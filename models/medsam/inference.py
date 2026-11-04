# -*- coding: utf-8 -*-

"""Inference / evaluation for optimized MedSAM multi-class segmentation.

Features:
- loads checkpoints from train_multi_gpus_v4.py and earlier versions
- optional horizontal-flip TTA
- optional automatic two-stage refinement (full frame -> predicted foreground crop)
- saves prediction masks / overlays / improved error maps
- exports per-split summary.json, class_metrics.csv, per_image_metrics.csv,
  per_class_summary.csv (with bootstrap CI), aggregate_summary.json
- Bootstrap 95% CI (per-class + mean fg)  ← NEW (ported from EoMT inference.py)
- MetricsAggregator class                  ← NEW (ported from EoMT inference.py)
- EoMT-style overlay: FG-only alpha blend  ← NEW
- EoMT colour palette                      ← NEW
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry
from segment_anything.modeling.multi_class_mask_decoder import MultiClassMaskDecoder

join = os.path.join

IMAGE_DIR_CANDIDATES = ("images", "imgs")
MASK_DIR_CANDIDATES = ("masks", "gts")

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
    0: (0,   0,   0),        # background – 검정
    1: (0,   255, 255),      # IEV        – 청록
    2: (0,   255, 0),        # Pubic arch – 초록
    3: (147, 112, 219),      # Spermatic cord – 보라
    4: (0,   0,   255),      # Vas deferens   – 파랑
}

SURFACE_STRUCTURE = generate_binary_structure(2, 1)


# ---------------------------------------------------------------
# 시각화 유틸  (EoMT 방식: FG-only overlay)
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
    h, w = pred_mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    correct_fg = (pred_mask == gt_mask) & (gt_mask > 0)
    fp        = (gt_mask == 0)  & (pred_mask > 0)
    miss      = (gt_mask > 0)  & (pred_mask == 0)
    wrong_fg  = (gt_mask > 0)  & (pred_mask > 0) & (pred_mask != gt_mask)
    out[correct_fg] = [0,   255, 0]
    out[fp]         = [255, 0,   0]
    out[miss]       = [0,   0,   255]
    out[wrong_fg]   = [255, 255, 0]
    return out


def resize_index_mask_nearest(mask: np.ndarray,
                               target_hw: Sequence[int]) -> np.ndarray:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if mask.shape == (target_h, target_w):
        return mask.astype(np.int64)
    max_label = int(mask.max()) if mask.size > 0 else 0
    mask_dtype = np.uint16 if max_label > 255 else np.uint8
    resized = Image.fromarray(mask.astype(mask_dtype)).resize(
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
    lower = float(np.percentile(boot_means, alpha))
    upper = float(np.percentile(boot_means, 100.0 - alpha))
    return lower, upper


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
    pred_surface = extract_surface(pred_binary)
    gt_surface   = extract_surface(gt_binary)
    pred_count, gt_count = int(pred_surface.sum()), int(gt_surface.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0
    dt_to_gt   = distance_transform_edt(~gt_surface)
    dt_to_pred = distance_transform_edt(~pred_surface)
    pred_close = int((dt_to_gt[pred_surface]   <= float(tau)).sum())
    gt_close   = int((dt_to_pred[gt_surface]   <= float(tau)).sum())
    return float((pred_close + gt_close) / max(pred_count + gt_count, 1))


def compute_binary_hd95(pred_binary: np.ndarray,
                        gt_binary: np.ndarray) -> Optional[float]:
    pred_binary = np.asarray(pred_binary, dtype=bool)
    gt_binary   = np.asarray(gt_binary,   dtype=bool)
    pred_has, gt_has = bool(pred_binary.any()), bool(gt_binary.any())
    if not pred_has and not gt_has:
        return None
    if pred_has != gt_has:
        return None
    pred_surface = extract_surface(pred_binary)
    gt_surface   = extract_surface(gt_binary)
    if not pred_surface.any() or not gt_surface.any():
        return None
    dt_to_gt   = distance_transform_edt(~gt_surface)
    dt_to_pred = distance_transform_edt(~pred_surface)
    all_dist = np.concatenate([dt_to_gt[pred_surface], dt_to_pred[gt_surface]])
    return float(np.percentile(all_dist, 95)) if all_dist.size > 0 else None


def compute_confusion_matrix(pred_mask: torch.Tensor,
                              gt_mask: torch.Tensor,
                              num_classes: int) -> torch.Tensor:
    pred   = pred_mask.view(-1).long()
    target = gt_mask.view(-1).long()
    valid  = (target >= 0) & (target < num_classes)
    pred, target = pred[valid], target[valid]
    inds = num_classes * target + pred
    return torch.bincount(inds, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def metrics_from_confusion_matrix(confmat: torch.Tensor,
                                   eps: float = 1e-6) -> Dict:
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
    mean_dice_fg = dice[1:][present_fg].mean().item() if present_fg.any() else dice.mean().item()
    mean_iou_fg  = iou[1:][present_fg].mean().item()  if present_fg.any() else iou.mean().item()
    mean_dice_all = dice[present_all].mean().item() if present_all.any() else dice.mean().item()
    mean_iou_all  = iou[present_all].mean().item()  if present_all.any() else iou.mean().item()
    return {
        "pixel_acc":       pixel_acc.item(),
        "mean_dice_all":   mean_dice_all,
        "mean_dice_fg":    mean_dice_fg,
        "mean_iou_all":    mean_iou_all,
        "mean_iou_fg":     mean_iou_fg,
        "class_dice":      dice.cpu().tolist(),
        "class_iou":       iou.cpu().tolist(),
        "class_precision": precision.cpu().tolist(),
        "class_recall":    recall.cpu().tolist(),
        "class_specificity": specificity.cpu().tolist(),
    }


# ---------------------------------------------------------------
# MetricsAggregator  (ported from EoMT inference.py)
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
        self.per_image_rows:    List[Dict] = []
        self.class_metrics_rows: List[Dict] = []
        self.class_nsd_values:  List[List[float]] = [[] for _ in range(self.num_classes)]
        self.class_hd95_values: List[List[float]] = [[] for _ in range(self.num_classes)]

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

        # micro FG stats
        tp_fg   = conf_i[1:, 1:].diag().sum().item() if self.num_classes > 1 else conf_i.diag().sum().item()
        pred_fg = conf_i[:, 1:].sum().item()          if self.num_classes > 1 else conf_i.sum().item()
        gt_fg   = conf_i[1:, :].sum().item()          if self.num_classes > 1 else conf_i.sum().item()

        row: Dict = {
            "image_name":       image_name,
            "pixel_acc":        m["pixel_acc"],
            "mean_dice_all":    m["mean_dice_all"],
            "mean_dice_fg":     m["mean_dice_fg"],
            "mean_iou_all":     m["mean_iou_all"],
            "mean_iou_fg":      m["mean_iou_fg"],
            "fg_precision_micro": tp_fg / max(pred_fg, 1),
            "fg_recall_micro":    tp_fg / max(gt_fg,   1),
            "fg_dice_micro":   (2 * tp_fg) / max(pred_fg + gt_fg, 1),
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
        summary["mean_nsd_fg"]  = safe_mean([safe_mean(v) for v in self.class_nsd_values[1:]])
        summary["mean_hd95_fg"] = safe_mean([safe_mean(v) for v in self.class_hd95_values[1:]])
        summary["class_nsd"]    = [safe_mean(v) for v in self.class_nsd_values]
        summary["class_hd95"]   = [safe_mean(v) for v in self.class_hd95_values]

        # Bootstrap CI: mean dice fg
        dice_fg_vals = [r["mean_dice_fg"] for r in self.per_image_rows]
        lo, hi = bootstrap_ci(dice_fg_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)
        summary["macro_mean_dice_fg_ci_lower"] = lo
        summary["macro_mean_dice_fg_ci_upper"] = hi

        # Bootstrap CI: mean NSD fg
        nsd_fg_vals = [r["mean_nsd_fg"] for r in self.per_image_rows if r.get("mean_nsd_fg") is not None]
        lo, hi = bootstrap_ci(nsd_fg_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)
        summary["mean_nsd_fg_ci_lower"] = lo
        summary["mean_nsd_fg_ci_upper"] = hi

        # Bootstrap CI: mean HD95 fg
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
# 콘솔 출력  (ported from EoMT inference.py)
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

def write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------

@dataclass
class DatasetPaths:
    name: str
    root: str
    image_dir: str
    mask_dir: Optional[str]


class MedSAMSegmentor(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder  = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            if boxes.dim() == 2:
                boxes = boxes[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=boxes, masks=None,
            )
        low_res_logits, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )
        return F.interpolate(low_res_logits, size=image.shape[-2:],
                             mode="bilinear", align_corners=False)


def _resolve_existing_dir(root: str, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        path = join(root, name)
        if os.path.isdir(path):
            return path
    return None


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def parse_checkpoint_meta(checkpoint_obj, fallback_num_classes: Optional[int],
                           fallback_prompt_mode: str):
    if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj:
        ckpt_num_classes = checkpoint_obj.get("num_classes")
        ckpt_prompt_mode = checkpoint_obj.get("prompt_mode")
    else:
        ckpt_num_classes = None
        ckpt_prompt_mode = None
    num_classes  = int(ckpt_num_classes if ckpt_num_classes is not None
                       else (fallback_num_classes or len(CLASS_NAMES)))
    prompt_mode  = str(ckpt_prompt_mode if ckpt_prompt_mode is not None
                       else fallback_prompt_mode)
    return num_classes, prompt_mode


def resolve_class_names(num_classes: int,
                        provided: Optional[Sequence[str]]) -> List[str]:
    if provided and len(provided) == num_classes:
        return list(provided)
    if num_classes == len(CLASS_NAMES):
        return CLASS_NAMES.copy()
    return [f"class_{i}" for i in range(num_classes)]


def load_model(base_checkpoint: str, trained_model: str, model_type: str,
               num_classes: int, device: torch.device) -> nn.Module:
    sam_model = sam_model_registry[model_type](checkpoint=base_checkpoint)
    pretrained_decoder = sam_model.mask_decoder
    sam_model.mask_decoder = MultiClassMaskDecoder.from_pretrained_decoder(
        pretrained_decoder, num_classes=num_classes
    )
    model = MedSAMSegmentor(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    )
    ckpt       = torch.load(trained_model, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")
    return model.to(device).eval()


def get_paths_for_split(root: str, split: str) -> DatasetPaths:
    split_root = join(root, split)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Split root does not exist: {split_root}")
    image_dir = _resolve_existing_dir(split_root, IMAGE_DIR_CANDIDATES)
    if image_dir is None:
        raise FileNotFoundError(f"Could not find image directory under {split_root}")
    mask_dir = _resolve_existing_dir(split_root, MASK_DIR_CANDIDATES)
    return DatasetPaths(name=split, root=split_root,
                        image_dir=image_dir, mask_dir=mask_dir)


# ---------------------------------------------------------------
# Preprocessing / inference helpers
# ---------------------------------------------------------------

def load_index_mask(mask_path: str) -> np.ndarray:
    gt = np.asarray(Image.open(mask_path), dtype=np.int64)
    if gt.ndim == 3:
        if gt.shape[2] == 1:
            gt = gt[..., 0]
        else:
            raise ValueError(f"Expected single-channel mask, got {gt.shape}: {mask_path}")
    return gt


def preprocess_rgb_array(image_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    resized = np.asarray(
        Image.fromarray(image_rgb).resize((image_size, image_size), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return torch.tensor(np.transpose(resized, (2, 0, 1)),
                        dtype=torch.float32).unsqueeze(0)


def preprocess_image(img_path: str,
                     image_size: int) -> Tuple[torch.Tensor, Tuple[int, int], np.ndarray]:
    image_rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    h, w = image_rgb.shape[:2]
    tensor = preprocess_rgb_array(image_rgb, image_size)
    return tensor, (h, w), image_rgb


def build_prompt_box(image_size: int, prompt_mode: str,
                     mask_path: Optional[str] = None,
                     bbox_shift: int = 0) -> torch.Tensor:
    if prompt_mode == "full":
        box = np.array([0, 0, image_size - 1, image_size - 1], dtype=np.float32)
        return torch.tensor(box).unsqueeze(0)
    if prompt_mode != "gt_bbox":
        raise ValueError(f"Unsupported prompt_mode={prompt_mode}")
    if mask_path is None:
        raise ValueError("mask_path required for prompt_mode='gt_bbox'")
    gt = load_index_mask(mask_path)
    gt_r = np.asarray(
        Image.fromarray(gt.astype(np.uint8)).resize((image_size, image_size), Image.NEAREST),
        dtype=np.int64,
    )
    fg = gt_r > 0
    if not fg.any():
        return torch.tensor(
            np.array([0, 0, image_size - 1, image_size - 1], dtype=np.float32)
        ).unsqueeze(0)
    ys, xs = np.where(fg)
    x0 = max(0, xs.min() - bbox_shift)
    x1 = min(image_size - 1, xs.max() + bbox_shift)
    y0 = max(0, ys.min() - bbox_shift)
    y1 = min(image_size - 1, ys.max() + bbox_shift)
    return torch.tensor(
        np.array([x0, y0, x1, y1], dtype=np.float32)
    ).unsqueeze(0)


def predict_logits_single(model: nn.Module, image_tensor: torch.Tensor,
                           box_tensor: torch.Tensor, device: torch.device,
                           use_amp: bool, tta_hflip: bool) -> torch.Tensor:
    image_tensor = image_tensor.to(device)
    box_tensor   = box_tensor.to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(use_amp and device.type == "cuda")):
            logits = model(image_tensor, box_tensor)
        if tta_hflip:
            img_flip = torch.flip(image_tensor, dims=[3])
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                 enabled=(use_amp and device.type == "cuda")):
                logits_flip = model(img_flip, box_tensor)
            logits = 0.5 * (logits + torch.flip(logits_flip, dims=[3]))
    return logits.float().cpu()


def resize_logits_to_original(logits: torch.Tensor,
                               orig_hw: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(logits, size=tuple(int(x) for x in orig_hw),
                         mode="bilinear", align_corners=False)


def compute_fg_bbox_from_mask(mask: np.ndarray,
                               expand_ratio: float = 0.20
                               ) -> Optional[Tuple[int, int, int, int]]:
    fg = mask > 0
    if not fg.any():
        return None
    ys, xs = np.where(fg)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = max(1, y1 - y0 + 1), max(1, x1 - x0 + 1)
    cy, cx = int(round(h * expand_ratio)), int(round(w * expand_ratio))
    return x0 - cx, y0 - cy, x1 + 1 + cx, y1 + 1 + cy


def clamp_crop_box(box: Tuple[int, int, int, int],
                   img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, min(x0, img_w - 1));  y0 = max(0, min(y0, img_h - 1))
    x1 = max(x0 + 1, min(x1, img_w)); y1 = max(y0 + 1, min(y1, img_h))
    return x0, y0, x1, y1


def infer_one_image(
    model: nn.Module, image_rgb: np.ndarray, image_size: int,
    prompt_mode: str, device: torch.device, use_amp: bool, tta_hflip: bool,
    two_stage_refine: bool, refine_expand_ratio: float,
    refine_blend: float, min_fg_pixels_for_refine: int,
) -> Tuple[np.ndarray, torch.Tensor]:
    orig_h, orig_w = image_rgb.shape[:2]
    image_tensor = preprocess_rgb_array(image_rgb, image_size)
    if prompt_mode != "full":
        raise ValueError("infer_one_image supports prompt_mode='full' only")
    box_tensor = build_prompt_box(image_size=image_size, prompt_mode="full")
    stage1_logits = resize_logits_to_original(
        predict_logits_single(model, image_tensor, box_tensor, device, use_amp, tta_hflip),
        (orig_h, orig_w),
    )
    final_logits = stage1_logits.clone()

    if two_stage_refine:
        stage1_mask = torch.argmax(stage1_logits, dim=1)[0].numpy().astype(np.uint8)
        if int((stage1_mask > 0).sum()) >= min_fg_pixels_for_refine:
            crop_box = compute_fg_bbox_from_mask(stage1_mask, expand_ratio=refine_expand_ratio)
            if crop_box is not None:
                x0, y0, x1, y1 = clamp_crop_box(crop_box, orig_w, orig_h)
                crop_logits = resize_logits_to_original(
                    predict_logits_single(
                        model, preprocess_rgb_array(image_rgb[y0:y1, x0:x1], image_size),
                        build_prompt_box(image_size=image_size, prompt_mode="full"),
                        device, use_amp, tta_hflip,
                    ),
                    (int(y1 - y0), int(x1 - x0)),
                )
                blend = float(max(0.0, min(1.0, refine_blend)))
                final_logits[:, :, y0:y1, x0:x1] = (
                    (1.0 - blend) * final_logits[:, :, y0:y1, x0:x1]
                    + blend * crop_logits
                )

    pred_mask = torch.argmax(final_logits, dim=1)[0].numpy().astype(np.int64)
    return pred_mask, final_logits[0]


# ---------------------------------------------------------------
# 시각화 저장
# ---------------------------------------------------------------

def save_visuals(image_name: str, img_np: np.ndarray,
                 gt_mask: Optional[np.ndarray], pred_mask: np.ndarray,
                 save_dir: str,
                 save_pred_masks: bool, save_color: bool,
                 save_overlay: bool, save_error_map: bool):
    if save_pred_masks:
        d = join(save_dir, "pred_masks");  os.makedirs(d, exist_ok=True)
        Image.fromarray(pred_mask.astype(np.uint8)).save(join(d, image_name))
    if save_color:
        d = join(save_dir, "pred_color");  os.makedirs(d, exist_ok=True)
        Image.fromarray(colorize_mask(pred_mask, CLASS_TO_COLOR)).save(join(d, image_name))
    if save_overlay and img_np is not None:
        d = join(save_dir, "overlay");     os.makedirs(d, exist_ok=True)
        Image.fromarray(overlay_mask_on_image(img_np, pred_mask)).save(join(d, image_name))
    if save_error_map and gt_mask is not None:
        d = join(save_dir, "error_map");   os.makedirs(d, exist_ok=True)
        Image.fromarray(build_error_map(pred_mask, gt_mask)).save(join(d, image_name))


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",             type=str, required=True)
    parser.add_argument("--splits",           nargs="+", default=["int_val"])
    parser.add_argument("--base_checkpoint",  type=str, default="./work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--trained_model",    type=str, required=True)
    parser.add_argument("--model_type",       type=str, default="vit_b")
    parser.add_argument("--image_size",       type=int, default=1024)
    parser.add_argument("--num_classes",      type=int, default=5)
    parser.add_argument("--prompt_mode",      type=str, default="full",
                        choices=["full", "gt_bbox"])
    parser.add_argument("--bbox_shift",       type=int, default=0)
    parser.add_argument("--class_names",      nargs="*", default=None)
    parser.add_argument("--save_dir",         type=str, required=True)
    parser.add_argument("--save_pred_masks",  action="store_true")
    parser.add_argument("--save_color",       action="store_true")
    parser.add_argument("--save_overlay",     action="store_true")
    parser.add_argument("--save_error_map",   action="store_true")
    parser.add_argument("--use_amp",          action="store_true")
    parser.add_argument("--tta_hflip",        action="store_true")
    parser.add_argument("--two_stage_refine", action="store_true")
    parser.add_argument("--refine_expand_ratio",      type=float, default=0.20)
    parser.add_argument("--refine_blend",             type=float, default=0.60)
    parser.add_argument("--min_fg_pixels_for_refine", type=int,   default=512)
    parser.add_argument("--nsd_tau",     type=float, default=2.0)
    parser.add_argument("--n_bootstrap", type=int,   default=1000)
    parser.add_argument("--ci",          type=float, default=95.0)
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_obj = torch.load(args.trained_model, map_location="cpu")
    num_classes, prompt_mode = parse_checkpoint_meta(
        ckpt_obj,
        fallback_num_classes=(args.num_classes if args.num_classes > 0 else None),
        fallback_prompt_mode=args.prompt_mode,
    )
    class_names = resolve_class_names(num_classes, args.class_names)

    print(f"Checkpoint  : {args.trained_model}")
    print(f"num_classes : {num_classes}  |  prompt_mode : {prompt_mode}  |  device : {device}")
    print(f"TTA hflip={args.tta_hflip}  |  two_stage_refine={args.two_stage_refine}")

    model = load_model(
        base_checkpoint=args.base_checkpoint,
        trained_model=args.trained_model,
        model_type=args.model_type,
        num_classes=num_classes,
        device=device,
    )

    aggregate_rows: List[Dict] = []

    for split in args.splits:
        paths    = get_paths_for_split(args.root, split)
        save_dir = join(args.save_dir, split)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n=== Evaluating [{split}] ===")
        print(f"images : {paths.image_dir}")
        print(f"masks  : {paths.mask_dir}")

        image_files = sorted(glob.glob(join(paths.image_dir, "*.png")))
        if not image_files:
            raise RuntimeError(f"No images found under {paths.image_dir}")
        print(f"[{split}] {len(image_files)} images")

        agg = MetricsAggregator(
            num_classes=num_classes,
            class_names=class_names,
            nsd_tau=args.nsd_tau,
            n_bootstrap=args.n_bootstrap,
            ci=args.ci,
        )

        for idx, img_path in enumerate(image_files, start=1):
            name      = os.path.basename(img_path)
            mask_path = (
                join(paths.mask_dir, name)
                if paths.mask_dir and os.path.isfile(join(paths.mask_dir, name))
                else None
            )
            _, (orig_h, orig_w), image_rgb = preprocess_image(img_path, args.image_size)
            pred_mask, _ = infer_one_image(
                model=model, image_rgb=image_rgb, image_size=args.image_size,
                prompt_mode=prompt_mode, device=device, use_amp=args.use_amp,
                tta_hflip=args.tta_hflip, two_stage_refine=args.two_stage_refine,
                refine_expand_ratio=args.refine_expand_ratio,
                refine_blend=args.refine_blend,
                min_fg_pixels_for_refine=args.min_fg_pixels_for_refine,
            )

            gt_mask: Optional[np.ndarray] = None
            if mask_path is not None:
                gt_mask = load_index_mask(mask_path)
                if gt_mask.shape != pred_mask.shape:
                    gt_mask = resize_index_mask_nearest(gt_mask, pred_mask.shape)
                agg.update(gt_mask, pred_mask, image_name=name)

            save_visuals(
                image_name=name, img_np=image_rgb,
                gt_mask=gt_mask, pred_mask=pred_mask,
                save_dir=save_dir,
                save_pred_masks=args.save_pred_masks,
                save_color=args.save_color,
                save_overlay=args.save_overlay,
                save_error_map=args.save_error_map,
            )

            if idx % 50 == 0 or idx == len(image_files):
                print(f"  [{split}] {idx}/{len(image_files)}")

        print(f"\nComputing bootstrap CI (n={args.n_bootstrap})...")
        summary = {
            "split":         split,
            "trained_model": args.trained_model,
            "num_evaluated": len(agg.per_image_rows),
            "nsd_tau":       args.nsd_tau,
            "n_bootstrap":   args.n_bootstrap,
            "ci":            args.ci,
            "prompt_mode":   prompt_mode,
            "image_size":    args.image_size,
            "num_classes":   num_classes,
            "class_names":   class_names,
            "tta_hflip":     args.tta_hflip,
            "two_stage_refine":    args.two_stage_refine,
            "refine_expand_ratio": args.refine_expand_ratio,
            "refine_blend":        args.refine_blend,
            **agg.compute_summary(),
        }
        per_class_summary = agg.compute_per_class_summary()

        write_json(summary,                          join(save_dir, "summary.json"))
        write_csv(agg.per_image_rows,                join(save_dir, "medsam_per_image_metrics.csv"))
        write_csv(agg.class_metrics_rows,            join(save_dir, "class_metrics.csv"))
        write_csv(per_class_summary,                 join(save_dir, "per_class_summary.csv"))

        print_summary(per_class_summary, summary, split, args.ci)
        print(f"Saved → {save_dir}")
        aggregate_rows.append(summary)

    write_json({"splits": aggregate_rows},
               join(args.save_dir, "aggregate_summary.json"))
    print(f"\nSaved aggregate → {join(args.save_dir, 'aggregate_summary.json')}")


if __name__ == "__main__":
    main()
