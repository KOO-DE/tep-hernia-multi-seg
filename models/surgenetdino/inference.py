# -*- coding: utf-8 -*-
"""
inference.py

Usage:
    python inference.py \
        --config configs/dinov2/tep/semantic/eomt_large_504.yaml \
        --ckpt lightning_logs/version_0/checkpoints/best.ckpt \
        --split int_val \
        --output_dir results/int_val \
        --save_pred_masks \
        --save_overlay

    python inference.py \
        --config configs/dinov2/tep/semantic/eomt_large_504.yaml \
        --ckpt lightning_logs/version_0/checkpoints/best.ckpt \
        --split ext_val \
        --output_dir results/ext_val
"""

import argparse
import csv
import importlib
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from lightning import seed_everything
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")
seed_everything(0, verbose=False)

# ---------------------------------------------------------------
# TEP class 정보
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

CLASS_TO_COLOR = {
    0: (0, 0, 0),
    1: (231, 226, 203),
    2: (252, 155, 192),
    3: (176, 145, 131),
    4: (169, 117, 87),
}

SURFACE_STRUCTURE = generate_binary_structure(2, 1)


# ---------------------------------------------------------------
# 시각화 유틸
# ---------------------------------------------------------------

def colorize_mask(mask: np.ndarray, class_to_color: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls, color in class_to_color.items():
        out[mask == cls] = np.asarray(color, dtype=np.uint8)
    return out


def overlay_mask_on_image(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize_mask(mask, CLASS_TO_COLOR)
    out = (1.0 - alpha) * image_rgb.astype(np.float32) + alpha * color.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def build_error_map(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    h, w = pred_mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    correct_fg = (pred_mask == gt_mask) & (gt_mask > 0)
    fp = (gt_mask == 0) & (pred_mask > 0)
    miss = (gt_mask > 0) & (pred_mask == 0)
    wrong_fg = (gt_mask > 0) & (pred_mask > 0) & (pred_mask != gt_mask)
    out[correct_fg] = [0, 255, 0]
    out[fp] = [255, 0, 0]
    out[miss] = [0, 0, 255]
    out[wrong_fg] = [255, 255, 0]
    return out


def resize_index_mask_nearest(mask: np.ndarray, target_hw: Sequence[int]) -> np.ndarray:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if mask.shape == (target_h, target_w):
        return mask.astype(np.int64)
    max_label = int(mask.max()) if mask.size > 0 else 0
    mask_dtype = np.uint16 if max_label > 255 else np.uint8
    resized = Image.fromarray(mask.astype(mask_dtype)).resize((target_w, target_h), Image.NEAREST)
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
    """
    Bootstrapping 기반 신뢰구간 계산.
    values: per-image metric 값 리스트 (None 제외)
    반환: (lower, upper) CI
    """
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


def compute_binary_nsd(pred_binary: np.ndarray, gt_binary: np.ndarray, tau: float = 2.0) -> Optional[float]:
    pred_binary = np.asarray(pred_binary, dtype=bool)
    gt_binary = np.asarray(gt_binary, dtype=bool)
    pred_has, gt_has = bool(pred_binary.any()), bool(gt_binary.any())
    if not pred_has and not gt_has:
        return None
    if pred_has != gt_has:
        return 0.0
    pred_surface = extract_surface(pred_binary)
    gt_surface = extract_surface(gt_binary)
    pred_count, gt_count = int(pred_surface.sum()), int(gt_surface.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0
    dt_to_gt = distance_transform_edt(~gt_surface)
    dt_to_pred = distance_transform_edt(~pred_surface)
    pred_close = int((dt_to_gt[pred_surface] <= float(tau)).sum())
    gt_close = int((dt_to_pred[gt_surface] <= float(tau)).sum())
    return float((pred_close + gt_close) / max(pred_count + gt_count, 1))


def compute_binary_hd95(pred_binary: np.ndarray, gt_binary: np.ndarray) -> Optional[float]:
    pred_binary = np.asarray(pred_binary, dtype=bool)
    gt_binary = np.asarray(gt_binary, dtype=bool)
    pred_has, gt_has = bool(pred_binary.any()), bool(gt_binary.any())
    if not pred_has and not gt_has:
        return None
    if pred_has != gt_has:
        return None
    pred_surface = extract_surface(pred_binary)
    gt_surface = extract_surface(gt_binary)
    if not pred_surface.any() or not gt_surface.any():
        return None
    dt_to_gt = distance_transform_edt(~gt_surface)
    dt_to_pred = distance_transform_edt(~pred_surface)
    all_dist = np.concatenate([dt_to_gt[pred_surface], dt_to_pred[gt_surface]])
    return float(np.percentile(all_dist, 95)) if all_dist.size > 0 else None


def compute_confusion_matrix(pred_t, gt_t, num_classes):
    valid = (gt_t >= 0) & (gt_t < num_classes) & (pred_t >= 0) & (pred_t < num_classes)
    inds = gt_t[valid] * num_classes + pred_t[valid]
    return torch.bincount(inds, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(confmat: torch.Tensor, eps: float = 1e-6) -> Dict:
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
    pixel_acc = (tp.sum() + eps) / (confmat.sum() + eps)
    present_all = confmat.sum(dim=1) > 0
    present_fg = confmat.sum(dim=1)[1:] > 0 if confmat.shape[0] > 1 else present_all
    mean_dice_fg = dice[1:][present_fg].mean().item() if present_fg.any() else dice.mean().item()
    mean_iou_fg = iou[1:][present_fg].mean().item() if present_fg.any() else iou.mean().item()
    mean_dice_all = dice[present_all].mean().item() if present_all.any() else dice.mean().item()
    mean_iou_all = iou[present_all].mean().item() if present_all.any() else iou.mean().item()
    return {
        "pixel_acc": pixel_acc.item(),
        "mean_dice_all": mean_dice_all,
        "mean_dice_fg": mean_dice_fg,
        "mean_iou_all": mean_iou_all,
        "mean_iou_fg": mean_iou_fg,
        "class_dice": dice.cpu().tolist(),
        "class_iou": iou.cpu().tolist(),
        "class_precision": precision.cpu().tolist(),
        "class_recall": recall.cpu().tolist(),
        "class_specificity": specificity.cpu().tolist(),
    }


# ---------------------------------------------------------------
# MetricsAggregator
# ---------------------------------------------------------------

class MetricsAggregator:
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None,
                 nsd_tau: float = 2.0, n_bootstrap: int = 1000, ci: float = 95.0):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.nsd_tau = float(nsd_tau)
        self.n_bootstrap = n_bootstrap
        self.ci = ci
        self.reset()

    def reset(self):
        self.confmat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        self.per_image_rows: List[Dict] = []
        self.class_metrics_rows: List[Dict] = []
        self.class_nsd_values: List[List[float]] = [[] for _ in range(self.num_classes)]
        self.class_hd95_values: List[List[float]] = [[] for _ in range(self.num_classes)]

    def update(self, gt_mask: np.ndarray, pred_mask: np.ndarray, image_name: str):
        gt_mask = np.asarray(gt_mask, dtype=np.int64)
        pred_mask = np.asarray(pred_mask, dtype=np.int64)

        if gt_mask.shape != pred_mask.shape:
            gt_mask = resize_index_mask_nearest(gt_mask, pred_mask.shape)

        pred_t = torch.from_numpy(pred_mask).long()
        gt_t = torch.from_numpy(gt_mask).long()
        conf_i = compute_confusion_matrix(pred_t, gt_t, self.num_classes)
        self.confmat += conf_i
        m = metrics_from_confusion_matrix(conf_i)

        per_image_nsd_fg, per_image_hd95_fg = [], []
        row = {
            "image_name": image_name,
            "pixel_acc": m["pixel_acc"],
            "mean_dice_all": m["mean_dice_all"],
            "mean_dice_fg": m["mean_dice_fg"],
            "mean_iou_all": m["mean_iou_all"],
            "mean_iou_fg": m["mean_iou_fg"],
        }

        tp_fg = conf_i[1:, 1:].diag().sum().item() if self.num_classes > 1 else conf_i.diag().sum().item()
        pred_fg = conf_i[:, 1:].sum().item() if self.num_classes > 1 else conf_i.sum().item()
        gt_fg = conf_i[1:, :].sum().item() if self.num_classes > 1 else conf_i.sum().item()
        row.update({
            "fg_precision_micro": tp_fg / max(pred_fg, 1),
            "fg_recall_micro": tp_fg / max(gt_fg, 1),
            "fg_dice_micro": (2 * tp_fg) / max(pred_fg + gt_fg, 1),
        })

        for cls_idx, cls_name in enumerate(self.class_names):
            pred_bin = pred_mask == cls_idx
            gt_bin = gt_mask == cls_idx
            nsd = compute_binary_nsd(pred_bin, gt_bin, tau=self.nsd_tau)
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
                "image_name": image_name,
                "class_idx": cls_idx,
                "class_name": cls_name,
                "dice": m["class_dice"][cls_idx],
                "iou": m["class_iou"][cls_idx],
                "precision": m["class_precision"][cls_idx],
                "recall": m["class_recall"][cls_idx],
                "specificity": m["class_specificity"][cls_idx],
                "nsd_tau": self.nsd_tau,
                "nsd": "" if nsd is None else float(nsd),
                "hd95": "" if hd95 is None else float(hd95),
            })

        row.update({
            "mean_nsd_fg": safe_mean(per_image_nsd_fg),
            "mean_hd95_fg": safe_mean(per_image_hd95_fg),
            "nsd_tau": self.nsd_tau,
        })
        self.per_image_rows.append(row)

    def compute_summary(self) -> Dict:
        summary = metrics_from_confusion_matrix(self.confmat)
        if not self.per_image_rows:
            return summary

        summary["macro_mean_dice_fg"] = float(np.mean([r["mean_dice_fg"] for r in self.per_image_rows]))
        summary["macro_mean_iou_fg"] = float(np.mean([r["mean_iou_fg"] for r in self.per_image_rows]))
        summary["macro_fg_recall"] = float(np.mean([r["fg_recall_micro"] for r in self.per_image_rows]))
        summary["fail_rate"] = float(np.mean([float(r["mean_dice_fg"] < 0.10) for r in self.per_image_rows]))
        summary["mean_nsd_fg"] = safe_mean([safe_mean(v) for v in self.class_nsd_values[1:]])
        summary["mean_hd95_fg"] = safe_mean([safe_mean(v) for v in self.class_hd95_values[1:]])
        summary["class_nsd"] = [safe_mean(v) for v in self.class_nsd_values]
        summary["class_hd95"] = [safe_mean(v) for v in self.class_hd95_values]

        # Mean fg 전체 CI (per-image mean_dice_fg 기반 bootstrapping)
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
            cls_rows = [r for r in self.class_metrics_rows if r["class_idx"] == cls_idx]

            dice_vals = [r["dice"] for r in cls_rows]
            nsd_vals = [None if r["nsd"] == "" else r["nsd"] for r in cls_rows]
            hd95_vals = [None if r["hd95"] == "" else r["hd95"] for r in cls_rows]

            # bootstrapping CI
            dice_lo, dice_hi = bootstrap_ci(dice_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)
            nsd_lo, nsd_hi = bootstrap_ci(nsd_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)
            hd95_lo, hd95_hi = bootstrap_ci(hd95_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)

            rows.append({
                "class_idx": cls_idx,
                "class_name": cls_name,
                "dice_mean": safe_mean(dice_vals),
                "dice_std": safe_std(dice_vals),
                "dice_ci_lower": dice_lo,
                "dice_ci_upper": dice_hi,
                "iou_mean": safe_mean([r["iou"] for r in cls_rows]),
                "precision_mean": safe_mean([r["precision"] for r in cls_rows]),
                "recall_mean": safe_mean([r["recall"] for r in cls_rows]),
                "nsd_mean": safe_mean(nsd_vals),
                "nsd_std": safe_std(nsd_vals),
                "nsd_ci_lower": nsd_lo,
                "nsd_ci_upper": nsd_hi,
                "hd95_mean": safe_mean(hd95_vals),
                "hd95_std": safe_std(hd95_vals),
                "hd95_ci_lower": hd95_lo,
                "hd95_ci_upper": hd95_hi,
            })
        return rows


# ---------------------------------------------------------------
# EoMT model 로드
# ---------------------------------------------------------------

def load_model(config_path: str, ckpt_path: str, device: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    img_size = tuple(config["data"]["init_args"]["img_size"])
    num_classes = config["data"]["init_args"]["num_classes"]

    enc_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_cls = getattr(
        importlib.import_module(enc_cfg["class_path"].rsplit(".", 1)[0]),
        enc_cfg["class_path"].rsplit(".", 1)[1],
    )
    enc_init = {k: v for k, v in enc_cfg.get("init_args", {}).items() if k != "ckpt_path"}
    encoder = enc_cls(img_size=img_size, **enc_init)

    net_cfg = config["model"]["init_args"]["network"]
    net_cls = getattr(
        importlib.import_module(net_cfg["class_path"].rsplit(".", 1)[0]),
        net_cfg["class_path"].rsplit(".", 1)[1],
    )
    net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
    network = net_cls(masked_attn_enabled=False, num_classes=num_classes, encoder=encoder, **net_kwargs)

    lit_cfg = config["model"]
    lit_cls = getattr(
        importlib.import_module(lit_cfg["class_path"].rsplit(".", 1)[0]),
        lit_cfg["class_path"].rsplit(".", 1)[1],
    )
    model_kwargs = {
        k: v for k, v in lit_cfg["init_args"].items()
        if k not in ("network", "ckpt_path", "freeze_strategy", "num_classes", "img_size")
    }
    model = lit_cls(img_size=img_size, num_classes=num_classes, network=network,
                    ckpt_path=None, **model_kwargs)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    model = model.eval().to(device)
    print(f"Model loaded: {ckpt_path}")
    return model, img_size


# ---------------------------------------------------------------
# Inference
# ---------------------------------------------------------------

@torch.no_grad()
def infer_single(model, img_np: np.ndarray, img_size: tuple, device: str) -> np.ndarray:
    orig_h, orig_w = img_np.shape[:2]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    img_resized = F.interpolate(
        img_tensor.unsqueeze(0), size=img_size, mode="bilinear", align_corners=False
    )[0]
    img_batch = img_resized.unsqueeze(0).to(device)

    with torch.cuda.amp.autocast():
        mask_logits_list, class_logits_list = model(img_batch)

    mask_logits = F.interpolate(mask_logits_list[-1], img_size, mode="bilinear")
    logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits_list[-1])

    pred = logits[0].argmax(dim=0).cpu()
    pred = F.interpolate(
        pred.unsqueeze(0).unsqueeze(0).float(),
        size=(orig_h, orig_w),
        mode="nearest",
    )[0, 0].long().numpy().astype(np.int64)
    return pred


# ---------------------------------------------------------------
# 시각화 저장
# ---------------------------------------------------------------

def save_visuals(image_name, img_np, gt_mask, pred_mask, save_dir,
                 save_pred_masks, save_color, save_overlay, save_error_map):
    if save_pred_masks:
        d = save_dir / "pred_masks"
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred_mask.astype(np.uint8)).save(d / image_name)
    if save_color:
        d = save_dir / "pred_color"
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(colorize_mask(pred_mask, CLASS_TO_COLOR)).save(d / image_name)
    if save_overlay and img_np is not None:
        d = save_dir / "overlay"
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay_mask_on_image(img_np, pred_mask)).save(d / image_name)
    if save_error_map:
        d = save_dir / "error_map"
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(build_error_map(pred_mask, gt_mask)).save(d / image_name)


# ---------------------------------------------------------------
# 콘솔 출력
# ---------------------------------------------------------------

def print_summary(per_class_summary, summary, split, ci):
    print(f"\n{'='*80}")
    print(f"Results [{split}]  n={summary.get('num_evaluated', '?')}  (95% CI, bootstrap)")
    print(f"{'='*80}")
    print(f"{'Class':<30} {'Dice':>6}  {'95% CI':<20} {'HD95':>7}  {'95% CI':<16} {'NSD':>6}  {'95% CI'}")
    print(f"{'-'*80}")

    for row in per_class_summary:
        if row["class_idx"] == 0:
            continue
        dice  = f"{row['dice_mean']:.3f}" if row["dice_mean"] is not None else "  N/A"
        hd95  = f"{row['hd95_mean']:.1f}" if row["hd95_mean"] is not None else "   N/A"
        nsd   = f"{row['nsd_mean']:.3f}" if row["nsd_mean"] is not None else "  N/A"

        dice_ci = (f"[{row['dice_ci_lower']:.3f}, {row['dice_ci_upper']:.3f}]"
                   if row["dice_ci_lower"] is not None else "N/A")
        hd95_ci = (f"[{row['hd95_ci_lower']:.1f}, {row['hd95_ci_upper']:.1f}]"
                   if row["hd95_ci_lower"] is not None else "N/A")
        nsd_ci  = (f"[{row['nsd_ci_lower']:.3f}, {row['nsd_ci_upper']:.3f}]"
                   if row["nsd_ci_lower"] is not None else "N/A")

        print(f"{row['class_name']:<30} {dice:>6}  {dice_ci:<20} {hd95:>7}  {hd95_ci:<16} {nsd:>6}  {nsd_ci}")

    print(f"{'-'*80}")

    mean_dice = summary.get('macro_mean_dice_fg', 0)
    mean_hd95 = summary.get('mean_hd95_fg') or 0
    mean_nsd  = summary.get('mean_nsd_fg') or 0

    dice_lo = summary.get('macro_mean_dice_fg_ci_lower')
    dice_hi = summary.get('macro_mean_dice_fg_ci_upper')
    hd95_lo = summary.get('mean_hd95_fg_ci_lower')
    hd95_hi = summary.get('mean_hd95_fg_ci_upper')
    nsd_lo  = summary.get('mean_nsd_fg_ci_lower')
    nsd_hi  = summary.get('mean_nsd_fg_ci_upper')

    dice_ci = f"[{dice_lo:.3f}, {dice_hi:.3f}]" if dice_lo is not None else "N/A"
    hd95_ci = f"[{hd95_lo:.1f}, {hd95_hi:.1f}]" if hd95_lo is not None else "N/A"
    nsd_ci  = f"[{nsd_lo:.3f}, {nsd_hi:.3f}]" if nsd_lo is not None else "N/A"

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
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_path", default="./TEP_dataset/ro")
    parser.add_argument("--split", default="int_val",
                        choices=["int_val", "ext_val", "train", "tune"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--save_pred_masks", action="store_true")
    parser.add_argument("--save_color", action="store_true")
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--save_error_map", action="store_true")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="bootstrapping 반복 횟수 (default: 1000)")
    parser.add_argument("--ci", type=float, default=95.0,
                        help="신뢰구간 % (default: 95.0)")
    args = parser.parse_args()

    device = f"cuda:{args.device}"

    if args.data_path is None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        args.data_path = cfg["data"]["init_args"]["path"]

    model, img_size = load_model(args.config, args.ckpt, device)

    img_dir  = Path(args.data_path) / args.split / "images"
    mask_dir = Path(args.data_path) / args.split / "masks"
    img_paths = sorted(img_dir.glob("*.png"))
    print(f"[{args.split}] {len(img_paths)} images")

    save_dir = Path(args.output_dir)
    agg = MetricsAggregator(
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        nsd_tau=NSD_TAU,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
    )

    for img_path in tqdm(img_paths, desc=f"Inferencing [{args.split}]"):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            print(f"[WARN] mask not found: {img_path.name}")
            continue

        img_np = np.array(Image.open(img_path).convert("RGB"))
        gt_mask = np.array(Image.open(mask_path)).astype(np.int64)
        pred_mask = infer_single(model, img_np, img_size, device)

        agg.update(gt_mask, pred_mask, image_name=img_path.name)
        save_visuals(
            image_name=img_path.name,
            img_np=img_np,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            save_dir=save_dir,
            save_pred_masks=args.save_pred_masks,
            save_color=args.save_color,
            save_overlay=args.save_overlay,
            save_error_map=args.save_error_map,
        )

    print(f"\nComputing bootstrap CI (n={args.n_bootstrap})...")
    summary = {
        "split": args.split,
        "config": args.config,
        "ckpt": args.ckpt,
        "num_evaluated": len(agg.per_image_rows),
        "nsd_tau": NSD_TAU,
        "n_bootstrap": args.n_bootstrap,
        "ci": args.ci,
        **agg.compute_summary(),
    }
    per_class_summary = agg.compute_per_class_summary()

    write_json(summary, save_dir / "summary.json")
    write_csv(agg.per_image_rows, save_dir / "per_image_metrics.csv")
    write_csv(agg.class_metrics_rows, save_dir / "class_metrics.csv")
    write_csv(per_class_summary, save_dir / "per_class_summary.csv")

    print_summary(per_class_summary, summary, args.split, args.ci)
    print(f"\nSaved → {save_dir}")


if __name__ == "__main__":
    main()
