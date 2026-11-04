# -*- coding: utf-8 -*-

# Adapted from segment_anything/modeling/mask_decoder.py for the TEP segmentation study:
# predict all classes with one class token and one output MLP per class.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-class mask decoder initialized from a pretrained SAM/MedSAM decoder."""

from copy import deepcopy
from typing import List, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d


class MultiClassMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_classes: int,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_classes = num_classes

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.class_tokens = nn.Embedding(num_classes, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            activation(),
        )
        # keep channel width aligned with SAM's upscaler output
        self.output_upscaling[3] = nn.ConvTranspose2d(
            transformer_dim // 4,
            transformer_dim // 8,
            kernel_size=2,
            stride=2,
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(num_classes)
            ]
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, num_classes, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del multimask_output  # kept for interface compatibility
        return self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat(
            [self.iou_token.weight, self.class_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if image_embeddings.shape[0] == tokens.shape[0]:
            src = image_embeddings
        elif image_embeddings.shape[0] == 1:
            src = image_embeddings.repeat(tokens.shape[0], 1, 1, 1)
        else:
            raise ValueError(
                "image_embeddings batch size must either match prompt batch size "
                "or be 1 for prompt expansion. "
                f"Got image_embeddings.shape[0]={image_embeddings.shape[0]} and "
                f"tokens.shape[0]={tokens.shape[0]}."
            )

        if dense_prompt_embeddings.shape[0] == src.shape[0]:
            dense_prompt = dense_prompt_embeddings
        elif dense_prompt_embeddings.shape[0] == 1:
            dense_prompt = dense_prompt_embeddings.repeat(src.shape[0], 1, 1, 1)
        else:
            raise ValueError(
                "dense_prompt_embeddings batch size mismatch: "
                f"{dense_prompt_embeddings.shape[0]} vs {src.shape[0]}"
            )

        if image_pe.shape[0] == src.shape[0]:
            pos_src = image_pe
        elif image_pe.shape[0] == 1:
            pos_src = image_pe.repeat(src.shape[0], 1, 1, 1)
        else:
            raise ValueError(
                f"image_pe batch size mismatch: {image_pe.shape[0]} vs {src.shape[0]}"
            )

        src = src + dense_prompt
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        class_tokens_out = hs[:, 1 : (1 + self.num_classes), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_classes):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](class_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, self.num_classes, h, w
        )
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred

    @classmethod
    def from_pretrained_decoder(
        cls,
        pretrained_decoder: nn.Module,
        num_classes: int,
    ) -> "MultiClassMaskDecoder":
        decoder = cls(
            transformer_dim=pretrained_decoder.transformer_dim,
            transformer=deepcopy(pretrained_decoder.transformer),
            num_classes=num_classes,
            activation=nn.GELU,
            iou_head_depth=len(pretrained_decoder.iou_prediction_head.layers),
            iou_head_hidden_dim=pretrained_decoder.iou_prediction_head.layers[0].out_features,
        )

        with torch.no_grad():
            decoder.iou_token.weight.copy_(pretrained_decoder.iou_token.weight)
            decoder.output_upscaling.load_state_dict(
                deepcopy(pretrained_decoder.output_upscaling.state_dict())
            )

            src_mask_tokens = pretrained_decoder.mask_tokens.weight.data
            num_src_tokens = src_mask_tokens.shape[0]
            for class_idx in range(num_classes):
                src_idx = class_idx % num_src_tokens
                decoder.class_tokens.weight[class_idx].copy_(src_mask_tokens[src_idx])
                decoder.output_hypernetworks_mlps[class_idx].load_state_dict(
                    deepcopy(
                        pretrained_decoder.output_hypernetworks_mlps[src_idx].state_dict()
                    )
                )

            _copy_mlp_hidden_layers(
                decoder.iou_prediction_head,
                pretrained_decoder.iou_prediction_head,
            )
            _copy_mlp_output_rows_cyclic(
                decoder.iou_prediction_head,
                pretrained_decoder.iou_prediction_head,
            )

        return decoder


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x



def _copy_mlp_hidden_layers(dst: MLP, src: nn.Module) -> None:
    num_hidden_layers = min(len(dst.layers), len(src.layers)) - 1
    if num_hidden_layers <= 0:
        return
    for layer_idx in range(num_hidden_layers):
        dst.layers[layer_idx].weight.copy_(src.layers[layer_idx].weight)
        dst.layers[layer_idx].bias.copy_(src.layers[layer_idx].bias)



def _copy_mlp_output_rows_cyclic(dst: MLP, src: nn.Module) -> None:
    dst_last = dst.layers[-1]
    src_last = src.layers[-1]
    num_src_rows = src_last.weight.shape[0]
    for row_idx in range(dst_last.weight.shape[0]):
        src_idx = row_idx % num_src_rows
        dst_last.weight[row_idx].copy_(src_last.weight[src_idx])
        dst_last.bias[row_idx].copy_(src_last.bias[src_idx])
