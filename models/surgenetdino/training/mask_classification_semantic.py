# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule


class MaskClassificationSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        freeze_strategy: str = "full",
        # freeze_strategy 옵션:
        #   "full"        - backbone 전체 학습 (default, 기존 동작)
        #   "decoder"     - backbone 완전 고정, EoMT head만 학습
        #   "last2"       - backbone 마지막 2블록 + EoMT head만 학습
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.ignore_idx = ignore_idx
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = range(num_classes)

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        self.init_metrics_semantic(ignore_idx, self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1)

        # freeze strategy 적용 (weights 로드 이후에 실행)
        self._apply_freeze_strategy(freeze_strategy)

    def _apply_freeze_strategy(self, strategy: str):
        backbone = self.network.encoder.backbone

        if strategy == "full":
            # 기존 동작 그대로, 아무것도 freeze하지 않음
            return

        elif strategy == "decoder":
            # backbone 전체 freeze
            for param in backbone.parameters():
                param.requires_grad = False
            frozen = sum(p.numel() for p in backbone.parameters())
            print(f"[freeze_strategy=decoder] Frozen backbone: {frozen:,} params")

        elif strategy == "last2":
            # backbone 전체 freeze 후 마지막 2블록만 unfreeze
            for param in backbone.parameters():
                param.requires_grad = False

            blocks = backbone.blocks
            num_blocks = len(blocks)
            # 마지막 2블록 unfreeze
            for block in blocks[num_blocks - 2:]:
                for param in block.parameters():
                    param.requires_grad = True
            # backbone norm도 unfreeze
            if hasattr(backbone, "norm"):
                for param in backbone.norm.parameters():
                    param.requires_grad = True

            frozen = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
            trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            print(f"[freeze_strategy=last2] Frozen: {frozen:,} / Trainable backbone: {trainable:,} params")

        else:
            raise ValueError(f"Unknown freeze_strategy: {strategy}. Choose from 'full', 'decoder', 'last2'")

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def on_validation_end(self):
        self._on_eval_end_semantic("val")
