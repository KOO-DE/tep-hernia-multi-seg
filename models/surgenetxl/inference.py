# -*- coding: utf-8 -*-
"""
inference.py

Usage:
    python inference.py \
        --img_dir  TEP_dataset/ro/int_val/images \
        --mask_dir TEP_dataset/ro/int_val/masks \
        --checkpoint checkpoints/best.pth \
        --save_dir results/int_val \
        --image_size 256 \
        --save_pred_masks \
        --save_overlay
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from metaformer import MetaFormerFPN

# ---------------------------------------------------------------
# TEP class 정보 (EoMT 기준 색상으로 통일)
# ---------------------------------------------------------------

CLASS_NAMES = [
    "background",
    "inferior_epigastric_vessels",
    "pubic_arch",
    "spermatic_cord",
    "vas_deferens",
]
NUM_CLASSES = 5
NSD_TAU = 2.0

CLASS_TO_COLOR: Dict[int, Tuple[int, int, int]] = {
    0: (0,   0,   0),        # background     – 검정
    1: (0,   255, 255),      # IEV            – 청록
    2: (0,   255, 0),        # Pubic arch     – 초록
    3: (147, 112, 219),      # Spermatic cord – 보라
    4: (0,   0,   255),      # Vas deferens   – 파랑
}

SURFACE_STRUCTURE = generate_binary_structure(2, 1)


# ---------------------------------------------------------------
# 시각화 유틸 (EoMT 방식: FG-only overlay)
# ---------------------------------------------------------------

def colorize_mask(mask: np.ndarray,
                  class_to_color: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls, color in class_to_color.items():
        out[mask == cls] = np.asarray(color, dtype=np.uint8)
    return out


def overlay_mask_on_image(image_rgb: np.ndarray,
                           mask: np.ndarray,
                           alpha: float = 0.45) -> np.ndarray:
    """Background(cls=0)를 제외하고 foreground 클래스만 alpha-blend."""
    out = image_rgb.astype(np.float32).copy()
    for cls, color in CLASS_TO_COLOR.items():
        if cls == 0:
            continue
        fg = mask == cls
        if fg.any():
            color_arr = np.array(color, dtype=np.float32)
            out[fg] = (1.0 - alpha) * out[fg] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def build_error_map(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    out[(pred_mask == gt_mask) & (gt_mask > 0)]               = [0,   255, 0]
    out[(gt_mask == 0)  & (pred_mask > 0)]                    = [255, 0,   0]
    out[(gt_mask > 0)   & (pred_mask == 0)]                   = [0,   0,   255]
    out[(gt_mask > 0)   & (pred_mask > 0) & (pred_mask != gt_mask)] = [255, 255, 0]
    return out


def resize_index_mask_nearest(mask: np.ndarray,
                               target_hw: Sequence[int]) -> np.ndarray:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if mask.shape == (target_h, target_w):
        return mask.astype(np.int64)
    dtype   = np.uint16 if int(mask.max()) > 255 else np.uint8
    resized = Image.fromarray(mask.astype(dtype)).resize(
        (target_w, target_h), Image.NEAREST
    )
    return np.asarray(resized, dtype=np.int64)


# ---------------------------------------------------------------
# Metric 계산
# ---------------------------------------------------------------

def safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def safe_std(values: Sequence[Optional[float]]) -> Optional[float]:
    values = [float(v) for v in values if v is not None]
    return float(np.std(values)) if values else None


def bootstrap_ci(
    values: List[Optional[float]],
    n_bootstrap: int = 1000,
    ci: float = 95.0,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    vals = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(vals) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(vals, size=len(vals), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (100.0 - ci) / 2.0
    return float(np.percentile(boot_means, alpha)), float(np.percentile(boot_means, 100.0 - alpha))


def extract_surface(binary_mask: np.ndarray) -> np.ndarray:
    binary_mask = np.asarray(binary_mask, dtype=bool)
    if not binary_mask.any():
        return np.zeros_like(binary_mask, dtype=bool)
    eroded = binary_erosion(binary_mask, structure=SURFACE_STRUCTURE, border_value=0)
    return np.logical_xor(binary_mask, eroded)


def compute_binary_nsd(pred_binary: np.ndarray,
                       gt_binary: np.ndarray,
                       tau: float = 2.0) -> Optional[float]:
    pred_binary = np.asarray(pred_binary, dtype=bool)
    gt_binary   = np.asarray(gt_binary,   dtype=bool)
    pred_has, gt_has = bool(pred_binary.any()), bool(gt_binary.any())
    if not pred_has and not gt_has:
        return None
    if pred_has != gt_has:
        return 0.0
    pred_surf = extract_surface(pred_binary)
    gt_surf   = extract_surface(gt_binary)
    pc, gc    = int(pred_surf.sum()), int(gt_surf.sum())
    if pc == 0 and gc == 0:
        return 1.0
    if pc == 0 or gc == 0:
        return 0.0
    dt_gt   = distance_transform_edt(~gt_surf)
    dt_pred = distance_transform_edt(~pred_surf)
    pred_close = int((dt_gt[pred_surf]   <= tau).sum())
    gt_close   = int((dt_pred[gt_surf]   <= tau).sum())
    return float((pred_close + gt_close) / max(pc + gc, 1))


def compute_binary_hd95(pred_binary: np.ndarray,
                        gt_binary: np.ndarray) -> Optional[float]:
    pred_binary = np.asarray(pred_binary, dtype=bool)
    gt_binary   = np.asarray(gt_binary,   dtype=bool)
    pred_has, gt_has = bool(pred_binary.any()), bool(gt_binary.any())
    if not pred_has and not gt_has:
        return None
    if pred_has != gt_has:
        return None
    pred_surf = extract_surface(pred_binary)
    gt_surf   = extract_surface(gt_binary)
    if not pred_surf.any() or not gt_surf.any():
        return None
    dt_gt   = distance_transform_edt(~gt_surf)
    dt_pred = distance_transform_edt(~pred_surf)
    all_dist = np.concatenate([dt_gt[pred_surf], dt_pred[gt_surf]])
    return float(np.percentile(all_dist, 95)) if all_dist.size > 0 else None


def compute_confusion_matrix(pred_t, gt_t, num_classes):
    valid = (gt_t >= 0) & (gt_t < num_classes) & (pred_t >= 0) & (pred_t < num_classes)
    inds  = gt_t[valid] * num_classes + pred_t[valid]
    return torch.bincount(inds, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def metrics_from_confusion_matrix(confmat: torch.Tensor, eps: float = 1e-6) -> Dict:
    confmat = confmat.float()
    tp = torch.diag(confmat)
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp
    tn = confmat.sum() - (tp + fp + fn)
    dice        = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou         = (tp + eps) / (tp + fp + fn + eps)
    precision   = (tp + eps) / (tp + fp + eps)
    recall      = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    pixel_acc   = (tp.sum() + eps) / (confmat.sum() + eps)
    present_all = confmat.sum(dim=1) > 0
    present_fg  = confmat.sum(dim=1)[1:] > 0 if confmat.shape[0] > 1 else present_all
    mean_dice_fg  = dice[1:][present_fg].mean().item() if present_fg.any() else dice.mean().item()
    mean_iou_fg   = iou[1:][present_fg].mean().item()  if present_fg.any() else iou.mean().item()
    mean_dice_all = dice[present_all].mean().item() if present_all.any() else dice.mean().item()
    mean_iou_all  = iou[present_all].mean().item()  if present_all.any() else iou.mean().item()
    return {
        "pixel_acc":         pixel_acc.item(),
        "mean_dice_all":     mean_dice_all,
        "mean_dice_fg":      mean_dice_fg,
        "mean_iou_all":      mean_iou_all,
        "mean_iou_fg":       mean_iou_fg,
        "class_dice":        dice.cpu().tolist(),
        "class_iou":         iou.cpu().tolist(),
        "class_precision":   precision.cpu().tolist(),
        "class_recall":      recall.cpu().tolist(),
        "class_specificity": specificity.cpu().tolist(),
    }


# ---------------------------------------------------------------
# MetricsAggregator  (EoMT 기준 – bootstrap CI 추가)
# ---------------------------------------------------------------

class MetricsAggregator:
    def __init__(self, num_classes: int,
                 class_names: Optional[List[str]] = None,
                 nsd_tau: float = 2.0,
                 n_bootstrap: int = 1000,
                 ci: float = 95.0):
        self.num_classes  = num_classes
        self.class_names  = class_names or [f"class_{i}" for i in range(num_classes)]
        self.nsd_tau      = float(nsd_tau)
        self.n_bootstrap  = n_bootstrap
        self.ci           = ci
        self.reset()

    def reset(self):
        self.confmat = torch.zeros((self.num_classes, self.num_classes),
                                   dtype=torch.int64)
        self.per_image_rows:     List[Dict] = []
        self.class_metrics_rows: List[Dict] = []
        self.class_nsd_values:   List[List[float]] = [[] for _ in range(self.num_classes)]
        self.class_hd95_values:  List[List[float]] = [[] for _ in range(self.num_classes)]

    def update(self, gt_mask: np.ndarray, pred_mask: np.ndarray, image_name: str):
        gt_mask   = np.asarray(gt_mask,   dtype=np.int64)
        pred_mask = np.asarray(pred_mask, dtype=np.int64)
        if gt_mask.shape != pred_mask.shape:
            gt_mask = resize_index_mask_nearest(gt_mask, pred_mask.shape)

        pred_t = torch.from_numpy(pred_mask).long()
        gt_t   = torch.from_numpy(gt_mask).long()
        conf_i = compute_confusion_matrix(pred_t, gt_t, self.num_classes)
        self.confmat += conf_i
        m = metrics_from_confusion_matrix(conf_i)

        per_image_nsd_fg:  List[float] = []
        per_image_hd95_fg: List[float] = []

        tp_fg   = conf_i[1:, 1:].diag().sum().item() if self.num_classes > 1 else conf_i.diag().sum().item()
        pred_fg = conf_i[:, 1:].sum().item()          if self.num_classes > 1 else conf_i.sum().item()
        gt_fg   = conf_i[1:, :].sum().item()          if self.num_classes > 1 else conf_i.sum().item()

        row: Dict = {
            "image_name":         image_name,
            "pixel_acc":          m["pixel_acc"],
            "mean_dice_all":      m["mean_dice_all"],
            "mean_dice_fg":       m["mean_dice_fg"],
            "mean_iou_all":       m["mean_iou_all"],
            "mean_iou_fg":        m["mean_iou_fg"],
            "fg_precision_micro": tp_fg / max(pred_fg, 1),
            "fg_recall_micro":    tp_fg / max(gt_fg,   1),
            "fg_dice_micro":     (2 * tp_fg) / max(pred_fg + gt_fg, 1),
        }

        for cls_idx, cls_name in enumerate(self.class_names):
            pred_bin = pred_mask == cls_idx
            gt_bin   = gt_mask   == cls_idx
            nsd  = compute_binary_nsd(pred_bin, gt_bin, tau=self.nsd_tau)
            hd95 = compute_binary_hd95(pred_bin, gt_bin)

            if nsd is not None:
                self.class_nsd_values[cls_idx].append(float(nsd))
                if cls_idx > 0:
                    per_image_nsd_fg.append(float(nsd))
            if hd95 is not None:
                self.class_hd95_values[cls_idx].append(float(hd95))
                if cls_idx > 0:
                    per_image_hd95_fg.append(float(hd95))

            self.class_metrics_rows.append({
                "image_name":  image_name,
                "class_idx":   cls_idx,
                "class_name":  cls_name,
                "dice":        m["class_dice"][cls_idx],
                "iou":         m["class_iou"][cls_idx],
                "precision":   m["class_precision"][cls_idx],
                "recall":      m["class_recall"][cls_idx],
                "specificity": m["class_specificity"][cls_idx],
                "nsd_tau":     self.nsd_tau,
                "nsd":         "" if nsd  is None else float(nsd),
                "hd95":        "" if hd95 is None else float(hd95),
            })

        row.update({
            "mean_nsd_fg":  safe_mean(per_image_nsd_fg),
            "mean_hd95_fg": safe_mean(per_image_hd95_fg),
            "nsd_tau":      self.nsd_tau,
        })
        self.per_image_rows.append(row)

    def compute_summary(self) -> Dict:
        summary = metrics_from_confusion_matrix(self.confmat)
        if not self.per_image_rows:
            return summary

        summary["macro_mean_dice_fg"] = float(np.mean([r["mean_dice_fg"] for r in self.per_image_rows]))
        summary["macro_mean_iou_fg"]  = float(np.mean([r["mean_iou_fg"]  for r in self.per_image_rows]))
        summary["macro_fg_recall"]    = float(np.mean([r["fg_recall_micro"] for r in self.per_image_rows]))
        summary["fail_rate"]          = float(np.mean([float(r["mean_dice_fg"] < 0.10) for r in self.per_image_rows]))
        summary["mean_nsd_fg"]   = safe_mean([safe_mean(v) for v in self.class_nsd_values[1:]])
        summary["mean_hd95_fg"]  = safe_mean([safe_mean(v) for v in self.class_hd95_values[1:]])
        summary["class_nsd"]     = [safe_mean(v) for v in self.class_nsd_values]
        summary["class_hd95"]    = [safe_mean(v) for v in self.class_hd95_values]

        dice_fg_vals = [r["mean_dice_fg"] for r in self.per_image_rows]
        lo, hi = bootstrap_ci(dice_fg_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)
        summary["macro_mean_dice_fg_ci_lower"] = lo
        summary["macro_mean_dice_fg_ci_upper"] = hi

        nsd_fg_vals = [r["mean_nsd_fg"] for r in self.per_image_rows if r.get("mean_nsd_fg") is not None]
        lo, hi = bootstrap_ci(nsd_fg_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)
        summary["mean_nsd_fg_ci_lower"] = lo
        summary["mean_nsd_fg_ci_upper"] = hi

        hd95_fg_vals = [r["mean_hd95_fg"] for r in self.per_image_rows if r.get("mean_hd95_fg") is not None]
        lo, hi = bootstrap_ci(hd95_fg_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)
        summary["mean_hd95_fg_ci_lower"] = lo
        summary["mean_hd95_fg_ci_upper"] = hi

        return summary

    def compute_per_class_summary(self) -> List[Dict]:
        rows = []
        for cls_idx, cls_name in enumerate(self.class_names):
            cls_rows  = [r for r in self.class_metrics_rows if r["class_idx"] == cls_idx]
            dice_vals = [r["dice"] for r in cls_rows]
            nsd_vals  = [None if r["nsd"]  == "" else r["nsd"]  for r in cls_rows]
            hd95_vals = [None if r["hd95"] == "" else r["hd95"] for r in cls_rows]

            dice_lo, dice_hi = bootstrap_ci(dice_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)
            nsd_lo,  nsd_hi  = bootstrap_ci(nsd_vals,  n_bootstrap=self.n_bootstrap, ci=self.ci)
            hd95_lo, hd95_hi = bootstrap_ci(hd95_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)

            rows.append({
                "class_idx":      cls_idx,
                "class_name":     cls_name,
                "dice_mean":      safe_mean(dice_vals),
                "dice_std":       safe_std(dice_vals),
                "dice_ci_lower":  dice_lo,
                "dice_ci_upper":  dice_hi,
                "iou_mean":       safe_mean([r["iou"]       for r in cls_rows]),
                "precision_mean": safe_mean([r["precision"] for r in cls_rows]),
                "recall_mean":    safe_mean([r["recall"]    for r in cls_rows]),
                "nsd_mean":       safe_mean(nsd_vals),
                "nsd_std":        safe_std(nsd_vals),
                "nsd_ci_lower":   nsd_lo,
                "nsd_ci_upper":   nsd_hi,
                "hd95_mean":      safe_mean(hd95_vals),
                "hd95_std":       safe_std(hd95_vals),
                "hd95_ci_lower":  hd95_lo,
                "hd95_ci_upper":  hd95_hi,
            })
        return rows


# ---------------------------------------------------------------
# 콘솔 출력  (EoMT 기준 – CI 포함)
# ---------------------------------------------------------------

def print_summary(per_class_summary: List[Dict], summary: Dict,
                  split: str, ci: float):
    print(f"\n{'='*80}")
    print(f"Results [{split}]  n={summary.get('num_evaluated', '?')}  ({ci}% CI, bootstrap)")
    print(f"{'='*80}")
    print(f"{'Class':<30} {'Dice':>6}  {'95% CI':<20} {'HD95':>7}  {'95% CI':<16} {'NSD':>6}  {'95% CI'}")
    print(f"{'-'*80}")
    for row in per_class_summary:
        if row["class_idx"] == 0:
            continue
        dice    = f"{row['dice_mean']:.3f}"  if row["dice_mean"]  is not None else "  N/A"
        hd95    = f"{row['hd95_mean']:.1f}"  if row["hd95_mean"]  is not None else "   N/A"
        nsd     = f"{row['nsd_mean']:.3f}"   if row["nsd_mean"]   is not None else "  N/A"
        dice_ci = (f"[{row['dice_ci_lower']:.3f}, {row['dice_ci_upper']:.3f}]"
                   if row["dice_ci_lower"] is not None else "N/A")
        hd95_ci = (f"[{row['hd95_ci_lower']:.1f}, {row['hd95_ci_upper']:.1f}]"
                   if row["hd95_ci_lower"] is not None else "N/A")
        nsd_ci  = (f"[{row['nsd_ci_lower']:.3f}, {row['nsd_ci_upper']:.3f}]"
                   if row["nsd_ci_lower"] is not None else "N/A")
        print(f"{row['class_name']:<30} {dice:>6}  {dice_ci:<20} {hd95:>7}  {hd95_ci:<16} {nsd:>6}  {nsd_ci}")
    print(f"{'-'*80}")
    mean_dice = summary.get("macro_mean_dice_fg", 0)
    mean_hd95 = summary.get("mean_hd95_fg") or 0
    mean_nsd  = summary.get("mean_nsd_fg")  or 0
    dice_lo = summary.get("macro_mean_dice_fg_ci_lower")
    dice_hi = summary.get("macro_mean_dice_fg_ci_upper")
    hd95_lo = summary.get("mean_hd95_fg_ci_lower")
    hd95_hi = summary.get("mean_hd95_fg_ci_upper")
    nsd_lo  = summary.get("mean_nsd_fg_ci_lower")
    nsd_hi  = summary.get("mean_nsd_fg_ci_upper")
    dice_ci = f"[{dice_lo:.3f}, {dice_hi:.3f}]" if dice_lo is not None else "N/A"
    hd95_ci = f"[{hd95_lo:.1f}, {hd95_hi:.1f}]"  if hd95_lo is not None else "N/A"
    nsd_ci  = f"[{nsd_lo:.3f}, {nsd_hi:.3f}]"    if nsd_lo  is not None else "N/A"
    print(f"{'Mean (excl. bg)':<30} {mean_dice:>6.3f}  {dice_ci:<20} {mean_hd95:>7.1f}  {hd95_ci:<16} {mean_nsd:>6.3f}  {nsd_ci}")
    print(f"Fail rate (Dice<0.1): {summary.get('fail_rate', 0)*100:.1f}%")
    print(f"{'='*80}")


# ---------------------------------------------------------------
# CSV / JSON 저장
# ---------------------------------------------------------------

def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------

class InferenceDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, image_size=256):
        self.img_dir    = img_dir
        self.mask_dir   = mask_dir
        self.image_size = image_size

        exts = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
        self.img_paths = []
        for ext in exts:
            self.img_paths.extend(glob.glob(os.path.join(img_dir, f"*.{ext}")))
        self.img_paths = sorted(self.img_paths)
        if not self.img_paths:
            raise ValueError(f"No images found in {img_dir}")

        self.has_gt = mask_dir is not None
        self.items  = []
        for img_path in self.img_paths:
            name      = os.path.basename(img_path)
            mask_path = None
            if self.has_gt:
                mask_path = os.path.join(mask_dir, name)
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
            self.items.append((img_path, mask_path))

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]
        name = os.path.basename(img_path)

        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        image_resized = cv2.resize(image_rgb, (self.image_size, self.image_size),
                                   interpolation=cv2.INTER_LINEAR)
        image_norm   = (image_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        image_tensor = torch.from_numpy(np.transpose(image_norm, (2, 0, 1))).float()

        sample = {
            "image":          image_tensor,
            "name":           name,
            "orig_h":         orig_h,
            "orig_w":         orig_w,
            "image_rgb_orig": image_rgb,
        }
        if self.has_gt:
            gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                raise ValueError(f"Failed to read mask: {mask_path}")
            sample["gt_mask"] = gt.astype(np.int64)
        return sample


def collate_fn(batch):
    result = {
        "image":          torch.stack([x["image"] for x in batch]),
        "name":           [x["name"] for x in batch],
        "orig_h":         [x["orig_h"] for x in batch],
        "orig_w":         [x["orig_w"] for x in batch],
        "image_rgb_orig": [x["image_rgb_orig"] for x in batch],
    }
    if "gt_mask" in batch[0]:
        result["gt_mask"] = [x["gt_mask"] for x in batch]
    return result


# ---------------------------------------------------------------
# Model 로드
# ---------------------------------------------------------------

def load_checkpoint_model(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    msg = model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint: {ckpt_path}\n{msg}")
    return model


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",    type=str, required=True)
    parser.add_argument("--mask_dir",   type=str, default="")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir",   type=str, required=True)
    parser.add_argument("--pretrained_ckpt",          type=str, default="SurgeNetXL")
    parser.add_argument("--load_pretrained_backbone", action="store_true")
    parser.add_argument("--num_classes",  type=int,   default=5)
    parser.add_argument("--image_size",   type=int,   default=512)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--save_pred_masks", action="store_true")
    parser.add_argument("--save_color",      action="store_true")
    parser.add_argument("--save_overlay",    action="store_true")
    parser.add_argument("--save_error_map",  action="store_true")
    parser.add_argument("--overlay_alpha",   type=float, default=0.45)
    parser.add_argument("--n_bootstrap",     type=int,   default=1000)
    parser.add_argument("--ci",              type=float, default=95.0)
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)

    ckpt_urls = {
        "SurgeNetXL":     "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetXL_checkpoint_epoch0050_teacher.pth?download=true",
        "SurgeNet":       "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_checkpoint_epoch0050_teacher.pth?download=true",
        "SurgeNet-Small": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetSmall_checkpoint_epoch0050_teacher.pth?download=true",
        "SurgeNet-Public":"https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/Public_checkpoint0050.pth?download=true",
    }
    pretrained_source = ckpt_urls.get(args.pretrained_ckpt, args.pretrained_ckpt)

    model = MetaFormerFPN(
        num_classes=args.num_classes,
        pretrained="SurgeNet",
        pretrained_weights=pretrained_source if args.load_pretrained_backbone else None,
        verbose=True,
    )
    model = load_checkpoint_model(model, args.checkpoint, device)
    model = model.to(device).eval()

    dataset = InferenceDataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir if args.mask_dir else None,
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=collate_fn)

    has_gt = bool(args.mask_dir)
    agg = (
        MetricsAggregator(
            num_classes=NUM_CLASSES,
            class_names=CLASS_NAMES,
            nsd_tau=NSD_TAU,
            n_bootstrap=args.n_bootstrap,
            ci=args.ci,
        ) if has_gt else None
    )

    # 저장 디렉토리 생성
    dirs: Dict[str, Path] = {}
    for key, flag in [
        ("pred_masks", args.save_pred_masks),
        ("pred_color", args.save_color),
        ("overlay",    args.save_overlay),
        ("error_map",  args.save_error_map),
    ]:
        if flag:
            d = save_dir / key
            d.mkdir(parents=True, exist_ok=True)
            dirs[key] = d

    for batch in tqdm(loader, desc="Inference"):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        preds  = torch.argmax(logits, dim=1).cpu().numpy()

        for i in range(len(batch["name"])):
            name      = batch["name"][i]
            orig_h    = batch["orig_h"][i]
            orig_w    = batch["orig_w"][i]
            image_rgb = batch["image_rgb_orig"][i]

            pred_mask = cv2.resize(
                preds[i].astype(np.uint8), (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)

            if "pred_masks" in dirs:
                Image.fromarray(pred_mask.astype(np.uint8)).save(dirs["pred_masks"] / name)
            if "pred_color" in dirs:
                Image.fromarray(colorize_mask(pred_mask, CLASS_TO_COLOR)).save(
                    dirs["pred_color"] / name
                )
            if "overlay" in dirs:
                Image.fromarray(
                    overlay_mask_on_image(image_rgb, pred_mask, alpha=args.overlay_alpha)
                ).save(dirs["overlay"] / name)

            if has_gt:
                gt_mask = batch["gt_mask"][i]
                if "error_map" in dirs:
                    gt_vis = (
                        resize_index_mask_nearest(gt_mask, pred_mask.shape)
                        if gt_mask.shape != pred_mask.shape else gt_mask
                    )
                    Image.fromarray(build_error_map(pred_mask, gt_vis)).save(
                        dirs["error_map"] / name
                    )
                agg.update(gt_mask, pred_mask, image_name=name)

    if has_gt and agg:
        print(f"\nComputing bootstrap CI (n={args.n_bootstrap})...")
        summary = {
            "img_dir":       args.img_dir,
            "checkpoint":    args.checkpoint,
            "num_evaluated": len(agg.per_image_rows),
            "nsd_tau":       NSD_TAU,
            "n_bootstrap":   args.n_bootstrap,
            "ci":            args.ci,
            **agg.compute_summary(),
        }
        per_class_summary = agg.compute_per_class_summary()

        write_json(summary,                   save_dir / "summary.json")
        write_csv(agg.per_image_rows,         save_dir / "surgenet_per_image_metrics.csv")
        write_csv(agg.class_metrics_rows,     save_dir / "class_metrics.csv")
        write_csv(per_class_summary,          save_dir / "per_class_summary.csv")

        print_summary(per_class_summary, summary,
                      Path(args.img_dir).parent.name, args.ci)
        print(f"\nSaved → {save_dir}")
    else:
        print(f"\nNo GT provided. Predictions saved → {save_dir}")


if __name__ == "__main__":
    main()
