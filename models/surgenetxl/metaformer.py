# Modified from TimJaspers0801/SurgeNet for the TEP segmentation study: support local checkpoints and nested checkpoint formats.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MetaFormer baselines including IdentityFormer, RandFormer, PoolFormerV2,
ConvFormer and CAFormer.
Some implementations are modified from timm (https://github.com/rwightman/pytorch-image-models).
"""

import math
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple
from timm.models.layers import trunc_normal_, DropPath


class MetaFormerFPN(nn.Module):
    def __init__(self, num_classes=1, pretrained='ImageNet', pretrained_weights=None, verbose=True):
        super().__init__()
        self.metaformer = caformer_s18(
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
            verbose=verbose,
        )

        feature_channels = (64, 128, 320, 512)

        self.FPN = FPN(
            encoder_channels=feature_channels,
            encoder_depth=3,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.0,
            merge_policy="cat",
            num_classes=num_classes,
            interpolation=4,
        )

    def forward(self, x):
        _, features = self.metaformer.forward_features(x)
        seg = self.FPN(*features)
        return seg

    def forward_features_list(self, x):
        _, features = self.metaformer.forward_features(x)
        return features


class Downsampling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        pre_norm=None,
        post_norm=None,
        pre_permute=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.post_norm(x)
        return x


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class SquaredReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.relu(x)
        return torch.mul(x, x)


class StarReLU(nn.Module):
    def __init__(
        self,
        scale_value=1.0,
        bias_value=0.0,
        scale_learnable=True,
        bias_learnable=True,
        mode=None,
        inplace=False,
    ):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=32,
        num_heads=None,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        proj_bias=False,
        **kwargs,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerNormGeneral(nn.Module):
    def __init__(
        self,
        affine_shape=None,
        normalized_dim=(3,),
        scale=True,
        bias=True,
        eps=1e-5,
    ):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        c2 = torch.mul(c, c)
        s = c2.mean(self.normalized_dim, keepdim=True)

        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class LayerNormWithoutBias(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )


class SepConv(nn.Module):
    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act1_layer=StarReLU,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4,
        out_features=None,
        act_layer=StarReLU,
        drop=0.0,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpHead(nn.Module):
    def __init__(
        self,
        dim,
        num_classes=1000,
        mlp_ratio=4,
        act_layer=SquaredReLU,
        norm_layer=nn.LayerNorm,
        head_dropout=0.0,
        bias=True,
    ):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):
    def __init__(
        self,
        dim,
        token_mixer=nn.Identity,
        mlp=Mlp,
        norm_layer=nn.LayerNorm,
        drop=0.0,
        drop_path=0.0,
        layer_scale_init_value=None,
        res_scale_init_value=None,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale1 = (
            Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        )
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale2 = (
            Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        )
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + self.layer_scale1(self.drop_path1(self.token_mixer(self.norm1(x))))
        x = self.res_scale2(x) + self.layer_scale2(self.drop_path2(self.mlp(self.norm2(x))))
        return x


DOWNSAMPLE_LAYERS_FOUR_STAGES = [
    partial(
        Downsampling,
        kernel_size=7,
        stride=4,
        padding=2,
        post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
    )
] + [
    partial(
        Downsampling,
        kernel_size=3,
        stride=2,
        padding=1,
        pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
        pre_permute=True,
    )
] * 3

# Adapted from: https://github.com/sail-sg/metaformer/tree/main
class MetaFormer(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
        token_mixers=nn.Identity,
        mlps=Mlp,
        norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
        drop_path_rate=0.0,
        head_dropout=0.0,
        layer_scale_init_values=None,
        res_scale_init_values=[None, None, 1.0, 1.0],
        output_norm=partial(nn.LayerNorm, eps=1e-6),
        head_fn=nn.Linear,
        pretrained_weights=None,
        verbose=True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.verbose = verbose

        if not isinstance(depths, (list, tuple)):
            depths = [depths]
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i + 1]) for i in range(num_stage)]
        )

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[
                    MetaFormerBlock(
                        dim=dims[i],
                        token_mixer=token_mixers[i],
                        mlp=mlps[i],
                        norm_layer=norm_layers[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_values[i],
                        res_scale_init_value=res_scale_init_values[i],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"norm"}

    def forward_features(self, x):
        feature_list = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feature_list.append(x.permute(0, 3, 1, 2))
        return self.norm(x.mean([1, 2])), feature_list

    def forward(self, x):
        x, _ = self.forward_features(x)
        x = self.head(x)
        return x

    def _unwrap_checkpoint(self, ckpt):
        if not isinstance(ckpt, dict):
            return ckpt

        for key in ["state_dict", "model", "teacher"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        return ckpt

    def _clean_state_dict(self, state_dict):
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            if new_key.startswith("backbone."):
                new_key = new_key[len("backbone."):]
            if new_key.startswith("head."):
                continue
            cleaned[new_key] = v
        return cleaned

    def load_pretrained_weights(self, source):
        if os.path.isfile(source):
            ckpt = torch.load(source, map_location="cpu")
        else:
            ckpt = torch.hub.load_state_dict_from_url(source, map_location="cpu", progress=self.verbose)

        state_dict = self._unwrap_checkpoint(ckpt)
        state_dict = self._clean_state_dict(state_dict)

        msg = self.load_state_dict(state_dict, strict=False)

        if self.verbose:
            print(f"Weights loaded from: {source}")
            print(f"Missing keys: {len(msg.missing_keys)}")
            print(f"Unexpected keys: {len(msg.unexpected_keys)}")
            if len(msg.missing_keys) > 0:
                print("Sample missing keys:", msg.missing_keys[:10])
            if len(msg.unexpected_keys) > 0:
                print("Sample unexpected keys:", msg.unexpected_keys[:10])


def caformer_s18(num_classes=0, pretrained='ImageNet', pretrained_weights=None, verbose=True, **kwargs):
    if pretrained == 'ImageNet':
        model = MetaFormer(
            in_chans=3,
            num_classes=num_classes,
            depths=[3, 3, 9, 3],
            dims=[64, 128, 320, 512],
            downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
            token_mixers=[SepConv, SepConv, Attention, Attention],
            mlps=Mlp,
            norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
            drop_path_rate=0.0,
            head_dropout=0.0,
            layer_scale_init_values=None,
            res_scale_init_values=[None, None, 1.0, 1.0],
            output_norm=partial(nn.LayerNorm, eps=1e-6),
            head_fn=MlpHead,
            pretrained_weights=pretrained_weights,
            verbose=verbose,
            **kwargs,
        )
    elif pretrained == 'SurgeNet':
        model = MetaFormer(
            in_chans=3,
            num_classes=num_classes,
            depths=[3, 3, 9, 3],
            dims=[64, 128, 320, 512],
            downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
            token_mixers=[
                partial(SepConv, act1_layer=nn.ReLU),
                partial(SepConv, act1_layer=nn.ReLU),
                Attention,
                Attention,
            ],
            mlps=partial(Mlp, act_layer=nn.ReLU),
            norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
            drop_path_rate=0.0,
            head_dropout=0.0,
            layer_scale_init_values=None,
            res_scale_init_values=[None, None, 1.0, 1.0],
            output_norm=partial(nn.LayerNorm, eps=1e-6),
            head_fn=MlpHead,
            pretrained_weights=pretrained_weights,
            verbose=verbose,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported pretrained type: {pretrained}")

    return model

# Code adapted from:
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/fpn/decoder.py
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/base/modules.py
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/base/heads.py
class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(f"`merge_policy` must be one of: ['add', 'cat'], got {policy}")
        self.policy = policy

    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError(f"`merge_policy` must be one of: ['add', 'cat'], got {self.policy}")


class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class SegmentationHead(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        activation=None,
        upsampling=1,
    ):
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class FPN(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=128,
        dropout=0.2,
        merge_policy="add",
        num_classes=1,
        interpolation=4,
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError(f"Encoder depth for FPN decoder cannot be less than 3, got {encoder_depth}.")

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(
                    pyramid_channels,
                    segmentation_channels,
                    n_upsamples=n_upsamples,
                )
                for n_upsamples in [3, 2, 1, 0]
            ]
        )

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

        self.segmentation_head = SegmentationHead(
            in_channels=self.out_channels,
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
            upsampling=interpolation,
        )

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        seg = self.segmentation_head(x)
        return seg


urls = {
    "ImageNet1k": "https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth",
    "SurgeNet-25epochs": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_checkpoint_epoch0025_teacher.pth?download=true",
    "SurgeNet-50epochs": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-Small": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetSmall_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-CHOLEC": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/CHOLEC_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-RAMIE": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RAMIE_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-RARP": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/RARP_checkpoint_epoch0050_teacher.pth?download=true",
    "SurgeNet-Public": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/Public_checkpoint0050.pth?download=true",
}


if __name__ == "__main__":
    model = caformer_s18(num_classes=4, pretrained='SurgeNet', pretrained_weights=urls['SurgeNet-50epochs'])
    dummy = torch.zeros([2, 3, 256, 256])
    out = model(dummy)
    print("Backbone output shape:", out.shape)

    model = MetaFormerFPN(num_classes=4, pretrained='SurgeNet', pretrained_weights=urls['SurgeNet-50epochs'])
    dummy = torch.zeros([2, 3, 256, 256])
    out = model(dummy)
    print("Segmentation output shape:", out.shape)
