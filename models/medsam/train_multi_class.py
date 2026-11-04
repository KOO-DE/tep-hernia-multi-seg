# -*- coding: utf-8 -*-

"""Optimized MedSAM multi-class training for TEP segmentation.

What is changed relative to the earlier train_multi_gpus version:
- uses pretrained decoder weights through MultiClassMaskDecoder.from_pretrained_decoder
- full-image prompt remains the default for automatic whole-image segmentation
- adds foreground-focused crop augmentation and realistic photometric augmentation
- uses dataset-derived class weights + weighted image sampling
- optimizes with weighted CE + generalized Dice + focal Tversky
- uses decoder-only warmup, then partial image encoder unfreezing
- uses cosine LR schedule, EMA, gradient clipping, and early stopping
- selects best checkpoint by macro per-image foreground Dice, not only pixel-weighted Dice
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
import math
import os
import random
import shutil
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from dataset import SurgicalIndexDataset
from segment_anything import sam_model_registry
from segment_anything.modeling.multi_class_mask_decoder import MultiClassMaskDecoder

join = os.path.join


# -----------------------------
# Reproducibility / DDP helpers
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(2024)
torch.backends.cudnn.benchmark = True



def reduce_tensor_sum(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        x = x.contiguous()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x



def reduce_tensor_mean(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        x = reduce_tensor_sum(x)
        x = x / dist.get_world_size()
    return x



def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


# -----------------------------
# Metrics / losses
# -----------------------------

def compute_confusion_matrix(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred = pred_mask.view(-1).long()
    target = gt_mask.view(-1).long()
    valid = (target >= 0) & (target < num_classes)
    pred = pred[valid]
    target = target[valid]
    inds = num_classes * target + pred
    confmat = torch.bincount(inds, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return confmat



def dice_iou_precision_recall_from_confmat(confmat: torch.Tensor, eps: float = 1e-6):
    confmat = confmat.float()
    tp = torch.diag(confmat)
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp
    tn = confmat.sum() - (tp + fp + fn)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    return dice, iou, precision, recall, specificity



def mean_present(values: torch.Tensor, present_mask: torch.Tensor) -> torch.Tensor:
    if present_mask.any():
        return values[present_mask].mean()
    return values.mean()



def pixel_accuracy_from_confusion_matrix(confmat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    confmat = confmat.float()
    return torch.diag(confmat).sum() / (confmat.sum() + eps)



def per_image_fg_dice_macro(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    # pred_mask/gt_mask: [B,H,W]
    bsz = pred_mask.shape[0]
    vals = []
    for b in range(bsz):
        pm = pred_mask[b]
        gm = gt_mask[b]
        class_dices = []
        for c in range(1, num_classes):
            gt_c = gm == c
            pred_c = pm == c
            if gt_c.any():
                inter = (gt_c & pred_c).sum().float()
                denom = gt_c.sum().float() + pred_c.sum().float()
                class_dices.append((2.0 * inter + eps) / (denom + eps))
        if len(class_dices) == 0:
            vals.append(torch.tensor(1.0, device=pred_mask.device))
        else:
            vals.append(torch.stack(class_dices).mean())
    return torch.stack(vals)



def generalized_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    if not include_background and num_classes > 1:
        probs = probs[:, 1:]
        one_hot = one_hot[:, 1:]

    target_sum = one_hot.sum(dim=(0, 2, 3))
    present = target_sum > 0
    if not present.any():
        return logits.sum() * 0.0

    probs = probs[:, present]
    one_hot = one_hot[:, present]
    target_sum = target_sum[present]

    w = 1.0 / (target_sum.square() + eps)
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    denom = (probs + one_hot).sum(dim=(0, 2, 3))
    dice = (2.0 * (w * inter).sum() + eps) / ((w * denom).sum() + eps)
    return 1.0 - dice



def focal_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    include_background: bool = False,
    class_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    if not include_background and num_classes > 1:
        probs = probs[:, 1:]
        one_hot = one_hot[:, 1:]
        if class_weights is not None:
            class_weights = class_weights[1:]

    target_sum = one_hot.sum(dim=(0, 2, 3))
    present = target_sum > 0
    if not present.any():
        return logits.sum() * 0.0

    probs = probs[:, present]
    one_hot = one_hot[:, present]
    if class_weights is not None:
        class_weights = class_weights[present]
        class_weights = class_weights / class_weights.sum().clamp_min(eps)

    tp = (probs * one_hot).sum(dim=(0, 2, 3))
    fp = (probs * (1.0 - one_hot)).sum(dim=(0, 2, 3))
    fn = ((1.0 - probs) * one_hot).sum(dim=(0, 2, 3))

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    loss = torch.pow(1.0 - tversky, gamma)

    if class_weights is not None:
        return (loss * class_weights).sum()
    return loss.mean()



def build_class_weights_from_pixel_counts(
    pixel_counts: np.ndarray,
    background_scale: float = 0.20,
) -> np.ndarray:
    pixel_counts = np.asarray(pixel_counts, dtype=np.float64)
    pixel_counts = np.maximum(pixel_counts, 1.0)
    freq = pixel_counts / pixel_counts.sum()
    weights = 1.0 / np.log(1.10 + freq)
    if len(weights) > 1:
        fg_ref = float(np.mean(weights[1:]))
        weights[0] = fg_ref * float(background_scale)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


# -----------------------------
# EMA + Scheduler
# -----------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            if torch.is_floating_point(tensor):
                self.shadow[name] = tensor.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for name, shadow_val in self.shadow.items():
            model_val = msd[name].detach()
            shadow_val.mul_(self.decay).add_(model_val, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        msd = model.state_dict()
        for name, shadow_val in self.shadow.items():
            self.backup[name] = msd[name].detach().clone()
            msd[name].copy_(shadow_val)

    def restore(self, model: nn.Module) -> None:
        if not self.backup:
            return
        msd = model.state_dict()
        for name, backup_val in self.backup.items():
            msd[name].copy_(backup_val)
        self.backup = {}


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, max_epochs: int, min_lr_ratio: float = 0.10):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.max_epochs = max(1, int(max_epochs))
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def register_new_group(self, lr: float) -> None:
        self.base_lrs.append(float(lr))

    def get_multiplier(self, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return float(epoch + 1) / float(self.warmup_epochs)
        if self.max_epochs <= self.warmup_epochs + 1:
            return 1.0
        progress = (epoch - self.warmup_epochs) / float(max(1, self.max_epochs - self.warmup_epochs - 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self, epoch: int) -> None:
        mult = self.get_multiplier(epoch)
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * mult


# -----------------------------
# Sampler
# -----------------------------
class DistributedWeightedSampler(Sampler[int]):
    def __init__(
        self,
        weights: np.ndarray,
        num_samples: int,
        replacement: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = bool(replacement)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = int(math.ceil(float(num_samples) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


# -----------------------------
# Model helpers
# -----------------------------

def configure_image_encoder_trainability(image_encoder, unfreeze_last_n: int):
    """
    Controls which parts of the image encoder are trainable.

    Modes
    -----
    unfreeze_last_n < 0:
        Full fine-tuning of the whole image encoder.

    unfreeze_last_n == 0:
        Fully freeze the whole image encoder (including neck).
        => decoder only tuning

    unfreeze_last_n > 0:
        Freeze encoder first, then unfreeze:
        - neck
        - last `unfreeze_last_n` blocks
        => partial fine-tuning
    """
    # freeze everything first
    for p in image_encoder.parameters():
        p.requires_grad = False

    # full fine-tuning
    if unfreeze_last_n < 0:
        for p in image_encoder.parameters():
            p.requires_grad = True
        return

    # true decoder-only tuning
    if unfreeze_last_n == 0:
        return

    # partial fine-tuning: neck + last n blocks
    if hasattr(image_encoder, "neck") and image_encoder.neck is not None:
        for p in image_encoder.neck.parameters():
            p.requires_grad = True

    if hasattr(image_encoder, "blocks") and image_encoder.blocks is not None:
        n_blocks = len(image_encoder.blocks)
        n = min(unfreeze_last_n, n_blocks)
        for blk in image_encoder.blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        mask_decoder: nn.Module,
        prompt_encoder: nn.Module,
        current_unfreeze_last_n: int,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

        configure_image_encoder_trainability(self.image_encoder, current_unfreeze_last_n)

    def set_unfreeze_last_n(self, n: int) -> None:
        configure_image_encoder_trainability(self.image_encoder, n)

    def forward(self, image: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            box_torch = boxes.float().to(image.device)
            if box_torch.dim() == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_logits, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )
        ori_res_logits = F.interpolate(low_res_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return ori_res_logits


# -----------------------------
# IO helpers
# -----------------------------

def atomic_torch_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".pth", dir=os.path.dirname(path))
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass



def safe_write_checkpoint(checkpoint, primary_path: str, fallback_path: Optional[str], is_main_host: bool = True):
    try:
        atomic_torch_save(checkpoint, primary_path)
        return primary_path
    except Exception as e:
        if is_main_host:
            print(f"[WARN] Failed to save checkpoint to {primary_path}: {e}", flush=True)
        if fallback_path is not None:
            try:
                atomic_torch_save(checkpoint, fallback_path)
                if is_main_host:
                    print(f"[INFO] Saved checkpoint to fallback path: {fallback_path}", flush=True)
                return fallback_path
            except Exception as e2:
                if is_main_host:
                    print(f"[WARN] Failed to save fallback checkpoint to {fallback_path}: {e2}", flush=True)
        return None



def save_curves(history: Dict[str, List[float]], save_path: str) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(20, 4.5))

    plt.subplot(1, 4, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(epochs, history["train_macro_dice_fg"], label="train")
    plt.plot(epochs, history["val_macro_dice_fg"], label="val")
    plt.title("Macro Image FG mDice")
    plt.xlabel("Epoch")
    plt.ylabel("mDice")
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(epochs, history["train_pixel_dice_fg"], label="train")
    plt.plot(epochs, history["val_pixel_dice_fg"], label="val")
    plt.title("Pixel-weighted FG mDice")
    plt.xlabel("Epoch")
    plt.ylabel("mDice")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(epochs, history["val_fail_rate_01"], label="val fail<0.1")
    plt.title("Catastrophic Failure Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Rate")
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()



def safe_write_curves(history, primary_path: str, fallback_path: Optional[str], is_main_host: bool = True):
    try:
        save_curves(history, primary_path)
        return primary_path
    except Exception as e:
        if is_main_host:
            print(f"[WARN] Failed to save curves to {primary_path}: {e}", flush=True)
        if fallback_path is not None:
            try:
                save_curves(history, fallback_path)
                if is_main_host:
                    print(f"[INFO] Saved curves to fallback path: {fallback_path}", flush=True)
                return fallback_path
            except Exception as e2:
                if is_main_host:
                    print(f"[WARN] Failed to save fallback curves to {fallback_path}: {e2}", flush=True)
        return None


# -----------------------------
# Training / evaluation
# -----------------------------
@torch.no_grad()
def maybe_apply_ema_for_eval(ema: Optional[ModelEMA], model_module: nn.Module):
    if ema is not None:
        ema.apply_shadow(model_module)


@torch.no_grad()
def maybe_restore_after_eval(ema: Optional[ModelEMA], model_module: nn.Module):
    if ema is not None:
        ema.restore(model_module)


@dataclass
class EpochMetrics:
    loss: float
    ce_loss: float
    gdice_loss: float
    tversky_loss: float
    pixel_acc: float
    pixel_dice_all: float
    pixel_dice_fg: float
    macro_dice_fg: float
    macro_recall_fg: float
    fail_rate_01: float
    class_dice: List[float]
    class_iou: List[float]
    class_precision: List[float]
    class_recall: List[float]
    class_specificity: List[float]



def run_epoch(
    model: nn.parallel.DistributedDataParallel,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    ce_loss: nn.Module,
    num_classes: int,
    gpu: int,
    train: bool,
    use_amp: bool,
    grad_acc_steps: int,
    ce_weight: float,
    gdice_weight: float,
    tversky_weight: float,
    tversky_alpha: float,
    tversky_beta: float,
    tversky_gamma: float,
    tversky_class_weights: Optional[torch.Tensor],
    include_background_in_overlap: bool,
    ema: Optional[ModelEMA],
    epoch: int,
    is_main_host: bool,
    log_interval: int,
    grad_clip_norm: float,
) -> EpochMetrics:
    model.train(train)

    loss_sum = torch.tensor(0.0, device=gpu)
    ce_sum = torch.tensor(0.0, device=gpu)
    gdice_sum = torch.tensor(0.0, device=gpu)
    tversky_sum = torch.tensor(0.0, device=gpu)
    sample_count = torch.tensor(0.0, device=gpu)
    confmat = torch.zeros((num_classes, num_classes), device=gpu, dtype=torch.float64)

    per_image_dice_sum = torch.tensor(0.0, device=gpu)
    per_image_recall_sum = torch.tensor(0.0, device=gpu)
    fail_count = torch.tensor(0.0, device=gpu)
    image_count = torch.tensor(0.0, device=gpu)

    num_batches = len(dataloader)
    if train:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)

    for step, (image, gt, boxes, _) in enumerate(dataloader):
        image = image.cuda(gpu, non_blocking=True)
        gt = gt.cuda(gpu, non_blocking=True)
        boxes = boxes.cuda(gpu, non_blocking=True)

        should_step = ((step + 1) % grad_acc_steps == 0) or ((step + 1) == num_batches)
        sync_context = model.no_sync if train and grad_acc_steps > 1 and not should_step else nullcontext

        with torch.set_grad_enabled(train):
            with sync_context():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    pred_logits = model(image, boxes)
                    loss_ce = ce_loss(pred_logits, gt)
                    loss_gdice = generalized_dice_loss(
                        pred_logits,
                        gt,
                        num_classes=num_classes,
                        include_background=include_background_in_overlap,
                    )
                    loss_tversky = focal_tversky_loss(
                        pred_logits,
                        gt,
                        num_classes=num_classes,
                        alpha=tversky_alpha,
                        beta=tversky_beta,
                        gamma=tversky_gamma,
                        include_background=include_background_in_overlap,
                        class_weights=tversky_class_weights,
                    )
                    loss = ce_weight * loss_ce + gdice_weight * loss_gdice + tversky_weight * loss_tversky

                if train:
                    loss_to_backward = loss / float(grad_acc_steps)
                    if scaler is not None and use_amp:
                        scaler.scale(loss_to_backward).backward()
                    else:
                        loss_to_backward.backward()

            if train and should_step:
                if grad_clip_norm > 0:
                    if scaler is not None and use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if scaler is not None and use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model.module)

        with torch.no_grad():
            pred_mask = torch.argmax(pred_logits, dim=1)
            batch_confmat = compute_confusion_matrix(pred_mask, gt, num_classes)
            bs = image.shape[0]
            loss_sum += loss.detach() * bs
            ce_sum += loss_ce.detach() * bs
            gdice_sum += loss_gdice.detach() * bs
            tversky_sum += loss_tversky.detach() * bs
            sample_count += bs
            confmat += batch_confmat.to(confmat.dtype)

            per_img_dice = per_image_fg_dice_macro(pred_mask, gt, num_classes=num_classes)
            per_image_dice_sum += per_img_dice.sum()
            fail_count += (per_img_dice < 0.10).float().sum()
            image_count += float(bs)

            # micro-like foreground recall per image, then macro over images
            for b in range(bs):
                conf_i = compute_confusion_matrix(pred_mask[b:b+1], gt[b:b+1], num_classes).float()
                tp = torch.diag(conf_i)[1:].sum()
                fn = conf_i[1:, :].sum() - torch.diag(conf_i)[1:].sum()
                rec = tp / (tp + fn + 1e-6)
                per_image_recall_sum += rec

        if is_main_host and log_interval > 0 and (((step + 1) % log_interval == 0) or ((step + 1) == num_batches)):
            phase = "train" if train else "val"
            print(
                f"[Epoch {epoch:03d}][{phase}] step {step + 1}/{num_batches} "
                f"loss={loss.detach().item():.4f} ce={loss_ce.detach().item():.4f} "
                f"gdice={loss_gdice.detach().item():.4f} tversky={loss_tversky.detach().item():.4f}",
                flush=True,
            )

    loss_sum = reduce_tensor_sum(loss_sum)
    ce_sum = reduce_tensor_sum(ce_sum)
    gdice_sum = reduce_tensor_sum(gdice_sum)
    tversky_sum = reduce_tensor_sum(tversky_sum)
    sample_count = reduce_tensor_sum(sample_count)
    confmat = reduce_tensor_sum(confmat)
    per_image_dice_sum = reduce_tensor_sum(per_image_dice_sum)
    per_image_recall_sum = reduce_tensor_sum(per_image_recall_sum)
    fail_count = reduce_tensor_sum(fail_count)
    image_count = reduce_tensor_sum(image_count)

    mean_loss = (loss_sum / sample_count).item()
    mean_ce = (ce_sum / sample_count).item()
    mean_gdice = (gdice_sum / sample_count).item()
    mean_tversky = (tversky_sum / sample_count).item()

    class_dice, class_iou, class_precision, class_recall, class_specificity = dice_iou_precision_recall_from_confmat(confmat)
    present_all = confmat.sum(dim=1) > 0
    present_fg = confmat.sum(dim=1)[1:] > 0 if num_classes > 1 else present_all
    pixel_dice_all = mean_present(class_dice, present_all).item()
    pixel_dice_fg = mean_present(class_dice[1:], present_fg).item() if num_classes > 1 else pixel_dice_all
    pixel_acc = pixel_accuracy_from_confusion_matrix(confmat).item()

    macro_dice_fg = (per_image_dice_sum / image_count.clamp_min(1.0)).item()
    macro_recall_fg = (per_image_recall_sum / image_count.clamp_min(1.0)).item()
    fail_rate_01 = (fail_count / image_count.clamp_min(1.0)).item()

    return EpochMetrics(
        loss=mean_loss,
        ce_loss=mean_ce,
        gdice_loss=mean_gdice,
        tversky_loss=mean_tversky,
        pixel_acc=pixel_acc,
        pixel_dice_all=pixel_dice_all,
        pixel_dice_fg=pixel_dice_fg,
        macro_dice_fg=macro_dice_fg,
        macro_recall_fg=macro_recall_fg,
        fail_rate_01=fail_rate_01,
        class_dice=class_dice.detach().cpu().tolist(),
        class_iou=class_iou.detach().cpu().tolist(),
        class_precision=class_precision.detach().cpu().tolist(),
        class_recall=class_recall.detach().cpu().tolist(),
        class_specificity=class_specificity.detach().cpu().tolist(),
    )



def sanity_check_one_batch(
    model: nn.parallel.DistributedDataParallel,
    dataloader: DataLoader,
    ce_loss: nn.Module,
    gpu: int,
    args,
    ce_weights_tensor: Optional[torch.Tensor],
    is_main_host: bool,
) -> None:
    if not is_main_host:
        return
    print("\n========== SANITY CHECK START ==========")
    model.train()
    image, gt, boxes, names = next(iter(dataloader))
    image = image.cuda(gpu, non_blocking=True)
    gt = gt.cuda(gpu, non_blocking=True)
    boxes = boxes.cuda(gpu, non_blocking=True)

    print(f"[SANITY] image shape: {tuple(image.shape)}")
    print(f"[SANITY] gt shape: {tuple(gt.shape)}")
    print(f"[SANITY] boxes shape: {tuple(boxes.shape)}")
    print(f"[SANITY] gt unique labels: {torch.unique(gt).detach().cpu().tolist()}")
    print(f"[SANITY] first sample name: {names[0]}")
    if ce_weights_tensor is not None:
        print(f"[SANITY] ce class weights: {ce_weights_tensor.detach().cpu().tolist()}")

    pred_logits = model(image, boxes)
    print(f"[SANITY] pred_logits shape: {tuple(pred_logits.shape)}")
    assert pred_logits.shape[1] == args.num_classes
    assert pred_logits.shape[2:] == gt.shape[1:]

    loss_ce = ce_loss(pred_logits, gt)
    loss_gdice = generalized_dice_loss(pred_logits, gt, args.num_classes, include_background=args.include_background_in_overlap)
    loss_tversky = focal_tversky_loss(
        pred_logits,
        gt,
        args.num_classes,
        alpha=args.tversky_alpha,
        beta=args.tversky_beta,
        gamma=args.tversky_gamma,
        include_background=args.include_background_in_overlap,
        class_weights=ce_weights_tensor,
    )
    loss = args.ce_weight * loss_ce + args.gdice_weight * loss_gdice + args.tversky_weight * loss_tversky
    loss.backward()

    grad_found = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(f"[SANITY] first grad found: {name}, grad_norm={param.grad.detach().norm().item():.6f}")
            grad_found = True
            break
    assert grad_found, "No gradients found after backward()"
    print("========== SANITY CHECK PASSED ==========\n")
    model.zero_grad(set_to_none=True)


# -----------------------------
# Optimizer utils
# -----------------------------

def build_optimizer(module: MedSAM, args) -> Tuple[torch.optim.Optimizer, Dict[str, int]]:
    decoder_params = [p for p in module.mask_decoder.parameters() if p.requires_grad]
    image_encoder_params = [p for p in module.image_encoder.parameters() if p.requires_grad]

    param_groups = []
    group_map = {}
    if decoder_params:
        group_map["decoder"] = len(param_groups)
        param_groups.append({"params": decoder_params, "lr": args.lr, "name": "decoder"})
    if image_encoder_params:
        group_map["image_encoder"] = len(param_groups)
        param_groups.append({"params": image_encoder_params, "lr": args.image_encoder_lr, "name": "image_encoder"})

    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    return optimizer, group_map



def maybe_add_image_encoder_param_group(
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    module: MedSAM,
    args,
    is_main_host: bool,
) -> None:
    image_encoder_params = [p for p in module.image_encoder.parameters() if p.requires_grad and not any(p is q for pg in optimizer.param_groups for q in pg["params"])]
    if not image_encoder_params:
        return
    optimizer.add_param_group({"params": image_encoder_params, "lr": args.image_encoder_lr, "name": "image_encoder"})
    scheduler.register_new_group(args.image_encoder_lr)
    if is_main_host:
        print(f"[INFO] Added image encoder param group with lr={args.image_encoder_lr}", flush=True)


# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train_root", type=str, default="TEP_dataset/ro/train")
parser.add_argument("--val_root", type=str, default="TEP_dataset/ro/tune")
parser.add_argument("--task_name", type=str, default="MedSAM_last2_decoder")
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth")
parser.add_argument("--work_dir", type=str, default="./work_dir")
parser.add_argument("--fallback_work_dir", type=str, default="/tmp/medsam_runs")

parser.add_argument("--num_epochs", type=int, default=120)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=2024)

parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--lr", type=float, default=8e-5)
parser.add_argument("--image_encoder_lr", type=float, default=5e-6)
parser.add_argument("--grad_clip_norm", type=float, default=1.0)
parser.add_argument("--use_amp", action="store_true")
parser.add_argument("--use_wandb", action="store_true")

parser.add_argument("--num_classes", type=int, default=5)
parser.add_argument("--image_size", type=int, default=1024)
parser.add_argument("--bbox_shift", type=int, default=20)
parser.add_argument("--img_suffix", type=str, default=".png")
parser.add_argument("--mask_suffix", type=str, default=".png")
parser.add_argument("--prompt_mode", type=str, default="full", choices=["full", "gt_bbox"])
parser.add_argument("--full_box_margin", type=int, default=0)

parser.add_argument("--decoder_only_warmup_epochs", type=int, default=0)
parser.add_argument("--unfreeze_image_encoder_last_n", type=int, default=2)
parser.add_argument("--disable_ema", action="store_true")
parser.add_argument("--ema_decay", type=float, default=0.999)
parser.add_argument("--warmup_epochs", type=int, default=5)
parser.add_argument("--min_lr_ratio", type=float, default=0.10)

parser.add_argument("--ce_weight", type=float, default=0.40)
parser.add_argument("--gdice_weight", type=float, default=0.20)
parser.add_argument("--tversky_weight", type=float, default=0.40)
parser.add_argument("--tversky_alpha", type=float, default=0.30)
parser.add_argument("--tversky_beta", type=float, default=0.70)
parser.add_argument("--tversky_gamma", type=float, default=0.75)
parser.add_argument("--include_background_in_overlap", action="store_true")
parser.add_argument("--disable_auto_class_weights", action="store_true")
parser.add_argument("--background_ce_scale", type=float, default=0.20)

parser.add_argument("--disable_weighted_sampler", action="store_true")
parser.add_argument("--weighted_sampler_power", type=float, default=1.0)
parser.add_argument("--weighted_sampler_bg_weight", type=float, default=0.30)

parser.add_argument("--roi_crop_prob", type=float, default=0.50)
parser.add_argument("--roi_context_min", type=float, default=0.15)
parser.add_argument("--roi_context_max", type=float, default=0.45)
parser.add_argument("--hflip_prob", type=float, default=0.50)
parser.add_argument("--brightness", type=float, default=0.15)
parser.add_argument("--contrast", type=float, default=0.15)
parser.add_argument("--saturation", type=float, default=0.10)
parser.add_argument("--gamma_min", type=float, default=0.90)
parser.add_argument("--gamma_max", type=float, default=1.10)
parser.add_argument("--blur_prob", type=float, default=0.10)
parser.add_argument("--noise_prob", type=float, default=0.10)

parser.add_argument("--world_size", type=int, default=0, help="0 => auto single-node world size")
parser.add_argument("--node_rank", type=int, default=0)
parser.add_argument("--master_addr", type=str, default="127.0.0.1")
parser.add_argument("--master_port", type=str, default="29500")
parser.add_argument("--bucket_cap_mb", type=int, default=25)
parser.add_argument("--grad_acc_steps", type=int, default=1)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--early_stop_patience", type=int, default=25)
parser.add_argument("--min_epochs_before_stop", type=int, default=20)

args = parser.parse_args()


# -----------------------------
# Main
# -----------------------------

def resolve_run_dirs(args):
    args.work_dir = os.path.abspath(args.work_dir)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    fallback_root = os.path.abspath(args.fallback_work_dir)
    fallback_save_path = join(fallback_root, args.task_name + "-" + run_id)
    return run_id, model_save_path, fallback_save_path



def main():
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node == 0:
        raise RuntimeError("No CUDA device found.")
    seed_everything(args.seed)

    run_id, model_save_path, fallback_save_path = resolve_run_dirs(args)
    args.run_id = run_id
    args.model_save_path = model_save_path
    args.fallback_save_path = fallback_save_path

    world_size = args.world_size if args.world_size > 0 else ngpus_per_node
    print(f"Spawning {ngpus_per_node} processes | effective world_size={world_size}", flush=True)
    print(f"Primary save dir: {args.model_save_path}", flush=True)
    print(f"Fallback save dir: {args.fallback_save_path}", flush=True)
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, world_size, args))



def main_worker(gpu: int, ngpus_per_node: int, world_size: int, args):
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))

    rank = int(args.node_rank) * ngpus_per_node + gpu
    is_main_host = rank == 0

    seed_everything(args.seed + rank)
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    model_save_path = args.model_save_path
    fallback_save_path = args.fallback_save_path
    run_id = args.run_id

    if is_main_host:
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(fallback_save_path, exist_ok=True)
        for src in [__file__, join(os.path.dirname(__file__), "dataset.py")]:
            try:
                shutil.copyfile(src, join(model_save_path, run_id + "_" + os.path.basename(src)))
            except Exception as e:
                print(f"[WARN] Could not copy {src} to primary save dir: {e}", flush=True)
    dist.barrier()

    wandb_run = None
    if args.use_wandb and is_main_host:
        import wandb
        wandb.login()
        wandb_run = wandb.init(project=args.task_name, config=vars(args))

    # Build datasets first: needed for class weights and weighted sampler.
    train_dataset = SurgicalIndexDataset(
        data_root=args.train_root,
        image_size=args.image_size,
        bbox_shift=args.bbox_shift,
        num_classes=args.num_classes,
        img_suffix=args.img_suffix,
        mask_suffix=args.mask_suffix,
        prompt_mode=args.prompt_mode,
        full_box_margin=args.full_box_margin,
        train=True,
        roi_crop_prob=args.roi_crop_prob,
        roi_context_min=args.roi_context_min,
        roi_context_max=args.roi_context_max,
        hflip_prob=args.hflip_prob,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        blur_prob=args.blur_prob,
        noise_prob=args.noise_prob,
    )
    val_dataset = SurgicalIndexDataset(
        data_root=args.val_root,
        image_size=args.image_size,
        bbox_shift=args.bbox_shift,
        num_classes=args.num_classes,
        img_suffix=args.img_suffix,
        mask_suffix=args.mask_suffix,
        prompt_mode=args.prompt_mode,
        full_box_margin=args.full_box_margin,
        train=False,
    )

    train_sampler: Sampler[int]
    if args.disable_weighted_sampler:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        if is_main_host:
            print("[INFO] Using standard DistributedSampler", flush=True)
    else:
        image_weights = train_dataset.get_image_sampling_weights(
            rare_power=args.weighted_sampler_power,
            background_weight=args.weighted_sampler_bg_weight,
        )
        train_sampler = DistributedWeightedSampler(
            weights=image_weights,
            num_samples=len(train_dataset),
            replacement=True,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
        )
        if is_main_host:
            print(f"[INFO] Using DistributedWeightedSampler | image weight range=({image_weights.min():.3f}, {image_weights.max():.3f})", flush=True)

    from torch.utils.data.distributed import DistributedSampler
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    # Build model.
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    pretrained_decoder = sam_model.mask_decoder
    sam_model.mask_decoder = MultiClassMaskDecoder.from_pretrained_decoder(pretrained_decoder, num_classes=args.num_classes)
    if hasattr(sam_model.mask_decoder, "iou_prediction_head"):
        for p in sam_model.mask_decoder.iou_prediction_head.parameters():
            p.requires_grad = False
        if is_main_host:
            print("[INFO] Froze iou_prediction_head (unused by segmentation loss)", flush=True)

    current_unfreeze = 0 if args.decoder_only_warmup_epochs > 0 else args.unfreeze_image_encoder_last_n
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        current_unfreeze_last_n=current_unfreeze,
    ).cuda(gpu)

    ema = None if args.disable_ema else ModelEMA(medsam_model, decay=args.ema_decay)

    ddp_model = nn.parallel.DistributedDataParallel(
        medsam_model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
        bucket_cap_mb=args.bucket_cap_mb,
    )

    optimizer, _ = build_optimizer(ddp_model.module, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.num_epochs, min_lr_ratio=args.min_lr_ratio)

    ce_weights_tensor = None
    if not args.disable_auto_class_weights:
        pixel_counts = train_dataset.get_class_pixel_counts()
        ce_weights = build_class_weights_from_pixel_counts(pixel_counts, background_scale=args.background_ce_scale)
        ce_weights_tensor = torch.tensor(ce_weights, dtype=torch.float32, device=gpu)
        if is_main_host:
            print(f"[INFO] train pixel counts: {pixel_counts.astype(int).tolist()}", flush=True)
            print(f"[INFO] CE class weights: {ce_weights.tolist()}", flush=True)
    ce_loss = nn.CrossEntropyLoss(weight=ce_weights_tensor)

    if is_main_host:
        total_params = sum(p.numel() for p in ddp_model.module.parameters())
        trainable_params = sum(p.numel() for p in ddp_model.module.parameters() if p.requires_grad)
        print(f"[Rank {rank}] total params: {total_params:,}", flush=True)
        print(f"[Rank {rank}] trainable params: {trainable_params:,}", flush=True)
        print(f"[Rank {rank}] train samples: {len(train_dataset)} | val samples: {len(val_dataset)}", flush=True)
        print(f"[INFO] current_unfreeze_last_n={current_unfreeze}", flush=True)

    sanity_check_one_batch(ddp_model, train_loader, ce_loss, gpu, args, ce_weights_tensor, is_main_host=is_main_host)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_macro_dice_fg": [],
        "val_macro_dice_fg": [],
        "train_pixel_dice_fg": [],
        "val_pixel_dice_fg": [],
        "val_fail_rate_01": [],
    }

    best_val_macro_dice = -1.0
    best_val_loss = float("inf")
    best_fail_rate = float("inf")
    best_pixel_acc = -1.0
    start_epoch = 0
    epochs_without_improve = 0

    if args.resume and os.path.isfile(args.resume):
        if is_main_host:
            print(f"=> loading checkpoint '{args.resume}'", flush=True)
        loc = f"cuda:{gpu}"
        checkpoint = torch.load(args.resume, map_location=loc)
        ddp_model.module.load_state_dict(strip_module_prefix(checkpoint["model"]), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None and args.use_amp:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_macro_dice = float(checkpoint.get("best_val_macro_dice_fg", best_val_macro_dice))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_fail_rate = float(checkpoint.get("best_fail_rate_01", best_fail_rate))
        best_pixel_acc = float(checkpoint.get("best_pixel_acc", best_pixel_acc))
        history = checkpoint.get("history", history)
        if checkpoint.get("ema_shadow") is not None and ema is not None:
            ema.shadow = {k: v.clone().to(gpu) for k, v in checkpoint["ema_shadow"].items()}
        if is_main_host:
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})", flush=True)
    dist.barrier()

    print("=== Trainable Parameters ===")
    for name, param in medsam_model.named_parameters():
        if param.requires_grad:
            print(name)
    print("============================")

    for epoch in range(start_epoch, args.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # Switch from decoder-only warmup to partial encoder fine-tuning.
        if epoch == args.decoder_only_warmup_epochs and args.decoder_only_warmup_epochs > 0:
            ddp_model.module.set_unfreeze_last_n(args.unfreeze_image_encoder_last_n)
            maybe_add_image_encoder_param_group(optimizer, scheduler, ddp_model.module, args, is_main_host=is_main_host)
            if is_main_host:
                trainable_params = sum(p.numel() for p in ddp_model.module.parameters() if p.requires_grad)
                print(f"[INFO] Switched to partial encoder fine-tuning | unfreeze_last_n={args.unfreeze_image_encoder_last_n} | trainable params={trainable_params:,}", flush=True)

        scheduler.step(epoch)

        train_metrics = run_epoch(
            model=ddp_model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            ce_loss=ce_loss,
            num_classes=args.num_classes,
            gpu=gpu,
            train=True,
            use_amp=args.use_amp,
            grad_acc_steps=args.grad_acc_steps,
            ce_weight=args.ce_weight,
            gdice_weight=args.gdice_weight,
            tversky_weight=args.tversky_weight,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            tversky_gamma=args.tversky_gamma,
            tversky_class_weights=ce_weights_tensor,
            include_background_in_overlap=args.include_background_in_overlap,
            ema=ema,
            epoch=epoch,
            is_main_host=is_main_host,
            log_interval=args.log_interval,
            grad_clip_norm=args.grad_clip_norm,
        )

        maybe_apply_ema_for_eval(ema, ddp_model.module)
        val_metrics = run_epoch(
            model=ddp_model,
            dataloader=val_loader,
            optimizer=None,
            scaler=None,
            ce_loss=ce_loss,
            num_classes=args.num_classes,
            gpu=gpu,
            train=False,
            use_amp=False,
            grad_acc_steps=1,
            ce_weight=args.ce_weight,
            gdice_weight=args.gdice_weight,
            tversky_weight=args.tversky_weight,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            tversky_gamma=args.tversky_gamma,
            tversky_class_weights=ce_weights_tensor,
            include_background_in_overlap=args.include_background_in_overlap,
            ema=None,
            epoch=epoch,
            is_main_host=is_main_host,
            log_interval=args.log_interval,
            grad_clip_norm=0.0,
        )
        maybe_restore_after_eval(ema, ddp_model.module)

        if is_main_host:
            history["train_loss"].append(train_metrics.loss)
            history["val_loss"].append(val_metrics.loss)
            history["train_macro_dice_fg"].append(train_metrics.macro_dice_fg)
            history["val_macro_dice_fg"].append(val_metrics.macro_dice_fg)
            history["train_pixel_dice_fg"].append(train_metrics.pixel_dice_fg)
            history["val_pixel_dice_fg"].append(val_metrics.pixel_dice_fg)
            history["val_fail_rate_01"].append(val_metrics.fail_rate_01)

            lrs = [pg["lr"] for pg in optimizer.param_groups]
            print(
                f"[Epoch {epoch:03d}] "
                f"LRs={['%.2e' % lr for lr in lrs]} | "
                f"Train loss={train_metrics.loss:.4f}, macro_fg={train_metrics.macro_dice_fg:.4f}, pixel_fg={train_metrics.pixel_dice_fg:.4f} | "
                f"Val loss={val_metrics.loss:.4f}, macro_fg={val_metrics.macro_dice_fg:.4f}, pixel_fg={val_metrics.pixel_dice_fg:.4f}, "
                f"pixel_fg={val_metrics.pixel_dice_fg:.4f}, pixel_acc={val_metrics.pixel_acc:.4f}, "
                f"macro_rec_fg={val_metrics.macro_recall_fg:.4f}, fail<0.1={val_metrics.fail_rate_01:.4f}",
                flush=True,
            )
            print(f"[Epoch {epoch:03d}] Val class Dice: {val_metrics.class_dice}", flush=True)
            print(f"[Epoch {epoch:03d}] Val class Recall: {val_metrics.class_recall}", flush=True)

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch,
                    "lr/decoder": optimizer.param_groups[0]["lr"],
                    "train/loss": train_metrics.loss,
                    "train/macro_dice_fg": train_metrics.macro_dice_fg,
                    "train/pixel_dice_fg": train_metrics.pixel_dice_fg,
                    "val/loss": val_metrics.loss,
                    "val/macro_dice_fg": val_metrics.macro_dice_fg,
                    "val/pixel_dice_fg": val_metrics.pixel_dice_fg,
                    "val/macro_recall_fg": val_metrics.macro_recall_fg,
                    "val/fail_rate_01": val_metrics.fail_rate_01,
                    "val/pixel_acc": val_metrics.pixel_acc,
                })

            checkpoint = {
                "model": ddp_model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if args.use_amp else None,
                "epoch": epoch,
                "train_metrics": train_metrics.__dict__,
                "val_metrics": val_metrics.__dict__,
                "history": history,
                "num_classes": args.num_classes,
                "prompt_mode": args.prompt_mode,
                "best_val_macro_dice_fg": best_val_macro_dice,
                "best_val_loss": best_val_loss,
                "best_fail_rate_01": best_fail_rate,
                "best_pixel_acc": best_pixel_acc,
                "current_unfreeze_last_n": args.unfreeze_image_encoder_last_n if epoch >= args.decoder_only_warmup_epochs else 0,
                "ce_class_weights": ce_weights_tensor.detach().cpu() if ce_weights_tensor is not None else None,
                "ema_shadow": {k: v.detach().cpu() for k, v in ema.shadow.items()} if ema is not None else None,
                "args": vars(args),
            }
            safe_write_checkpoint(checkpoint, join(model_save_path, "medsam_model_latest.pth"), join(fallback_save_path, "medsam_model_latest.pth"), is_main_host=True)

            improved = False
            if val_metrics.macro_dice_fg > best_val_macro_dice + 1e-5:
                best_val_macro_dice = val_metrics.macro_dice_fg
                checkpoint["best_val_macro_dice_fg"] = best_val_macro_dice
                safe_write_checkpoint(checkpoint, join(model_save_path, "medsam_model_best_macro_dice.pth"), join(fallback_save_path, "medsam_model_best_macro_dice.pth"), is_main_host=True)
                improved = True

            if val_metrics.loss < best_val_loss - 1e-5:
                best_val_loss = val_metrics.loss
                checkpoint["best_val_loss"] = best_val_loss
                safe_write_checkpoint(checkpoint, join(model_save_path, "medsam_model_best_loss.pth"), join(fallback_save_path, "medsam_model_best_loss.pth"), is_main_host=True)

            if val_metrics.fail_rate_01 < best_fail_rate - 1e-5:
                best_fail_rate = val_metrics.fail_rate_01
                checkpoint["best_fail_rate_01"] = best_fail_rate
                safe_write_checkpoint(checkpoint, join(model_save_path, "medsam_model_best_failrate.pth"), join(fallback_save_path, "medsam_model_best_failrate.pth"), is_main_host=True)
            
            if val_metrics.pixel_acc > best_pixel_acc + 1e-5:
                best_pixel_acc = val_metrics.pixel_acc
                checkpoint["best_pixel_acc"] = best_pixel_acc
                safe_write_checkpoint(
                    checkpoint,
                    join(model_save_path, "medsam_model_best_pixel_acc.pth"),
                    join(fallback_save_path, "medsam_model_best_pixel_acc.pth"),
                    is_main_host=True,
                )
            safe_write_curves(history, join(model_save_path, f"{args.task_name}_curves.png"), join(fallback_save_path, f"{args.task_name}_curves.png"), is_main_host=True)

            if improved:
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

        # sync early stop decision from rank0
        stop_tensor = torch.tensor(0, device=gpu, dtype=torch.int64)
        if is_main_host and epoch + 1 >= args.min_epochs_before_stop and epochs_without_improve >= args.early_stop_patience:
            stop_tensor.fill_(1)
            print(f"[INFO] Early stopping triggered at epoch {epoch} (no macro-dice improvement for {epochs_without_improve} epochs)", flush=True)
        dist.broadcast(stop_tensor, src=0)
        dist.barrier()
        if stop_tensor.item() == 1:
            break

    if wandb_run is not None:
        wandb_run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
