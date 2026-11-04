# -*- coding: utf-8 -*-
# Modified from bowang-lab/MedSAM for the TEP segmentation study: support nested checkpoints and non-strict loading.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from pathlib import Path
import urllib.request

import torch

from .modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


AUTO_DOWNLOAD_URLS = {
    "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
}


def _normalize_state_dict(state_dict):
    """Handle plain SAM/MedSAM checkpoints and common nested checkpoint formats."""
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(state_dict)}")

    for candidate_key in ("model", "state_dict"):
        if candidate_key in state_dict and isinstance(state_dict[candidate_key], dict):
            state_dict = state_dict[candidate_key]
            break

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    return state_dict



def _maybe_download_checkpoint(checkpoint_path: Path) -> None:
    if checkpoint_path.exists():
        return

    url = AUTO_DOWNLOAD_URLS.get(checkpoint_path.name)
    if url is None:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cmd = input(f"Download {checkpoint_path.name} from Facebook AI? [y]/n: ")
    if len(cmd) > 0 and cmd.lower() != "y":
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {checkpoint_path.name}...")
    urllib.request.urlretrieve(url, checkpoint_path)
    print(f"Downloaded {checkpoint_path.name}")



def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()

    if checkpoint is None:
        return sam

    checkpoint_path = Path(checkpoint)
    _maybe_download_checkpoint(checkpoint_path)

    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
    state_dict = _normalize_state_dict(state_dict)

    missing_keys, unexpected_keys = sam.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print("[build_sam] Missing keys:", missing_keys)
        print("[build_sam] Unexpected keys:", unexpected_keys)

    return sam
