# -*- coding: utf-8 -*-
"""
evaluate.py

Usage:
    python evaluate.py \
        --gt_dir path/to/gt_masks \
        --pred_dir path/to/pred_masks \
        --output_dir results/eval \
        --split int_val
"""

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# TEP class information
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

SURFACE_STRUCTURE = generate_binary_structure(2, 1)


# ---------------------------------------------------------------
# Metric computation
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


def compute_binary_nsd(pred_binary: np.ndarray, gt_binary: np.ndarray, tau: float = 2.0) -> Optional[float]:
    pred_binary = np.asarray(pred_binary, dtype=bool)
    gt_binary = np.asarray(gt_binary, dtype=bool)
    pred_has, gt_has = bool(pred_binary.any()), bool(gt_binary.any())
    if not pred_has and not gt_has:
        return None  # TN: excluded
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
        return None  # TN: excluded
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


def compute_binary_dice(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    """
    TN (no GT + no Pred): dice = 1.0
    FP (no GT + Pred present) or FN (GT present + no Pred): dice = 0.0
    """
    pred_binary = np.asarray(pred_binary, dtype=bool)
    gt_binary = np.asarray(gt_binary, dtype=bool)
    pred_has = bool(pred_binary.any())
    gt_has = bool(gt_binary.any())
    if not pred_has and not gt_has:
        return 1.0  # TN: correct
    tp = int((pred_binary & gt_binary).sum())
    fp = int((pred_binary & ~gt_binary).sum())
    fn = int((~pred_binary & gt_binary).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


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


def resize_index_mask_nearest(mask: np.ndarray, target_hw: Sequence[int]) -> np.ndarray:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if mask.shape == (target_h, target_w):
        return mask.astype(np.int64)
    max_label = int(mask.max()) if mask.size > 0 else 0
    mask_dtype = np.uint16 if max_label > 255 else np.uint8
    resized = Image.fromarray(mask.astype(mask_dtype)).resize((target_w, target_h), Image.NEAREST)
    return np.asarray(resized, dtype=np.int64)


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
        # per-class dice/nsd/hd95 values (with TN excluded)
        self.class_dice_values: List[List[float]] = [[] for _ in range(self.num_classes)]
        self.class_nsd_values: List[List[float]] = [[] for _ in range(self.num_classes)]
        self.class_hd95_values: List[List[float]] = [[] for _ in range(self.num_classes)]

    def update(self, gt_mask: np.ndarray, pred_mask: np.ndarray, image_name: str):
        gt_mask = np.asarray(gt_mask, dtype=np.int64)
        pred_mask = np.asarray(pred_mask, dtype=np.int64)
        if gt_mask.shape != pred_mask.shape:
            pred_mask = resize_index_mask_nearest(pred_mask, gt_mask.shape)

        pred_t = torch.from_numpy(pred_mask).long()
        gt_t = torch.from_numpy(gt_mask).long()
        conf_i = compute_confusion_matrix(pred_t, gt_t, self.num_classes)
        self.confmat += conf_i

        per_image_dice_fg, per_image_nsd_fg, per_image_hd95_fg = [], [], []

        for cls_idx, cls_name in enumerate(self.class_names):
            pred_bin = pred_mask == cls_idx
            gt_bin = gt_mask == cls_idx

            # Dice with TN excluded
            dice = compute_binary_dice(pred_bin, gt_bin)
            nsd = compute_binary_nsd(pred_bin, gt_bin, tau=self.nsd_tau)
            hd95 = compute_binary_hd95(pred_bin, gt_bin)

            self.class_dice_values[cls_idx].append(dice)
            if cls_idx > 0:
                per_image_dice_fg.append(dice)
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
                "dice": dice,
                "nsd_tau": self.nsd_tau,
                "nsd": "" if nsd is None else float(nsd),
                "hd95": "" if hd95 is None else float(hd95),
            })

        # per-image summary (fg only, TN excluded)
        row = {
            "image_name": image_name,
            "mean_dice_fg": safe_mean(per_image_dice_fg),
            "mean_nsd_fg": safe_mean(per_image_nsd_fg),
            "mean_hd95_fg": safe_mean(per_image_hd95_fg),
            "nsd_tau": self.nsd_tau,
        }
        self.per_image_rows.append(row)

    def compute_summary(self) -> Dict:
        summary = metrics_from_confusion_matrix(self.confmat)

        if not self.per_image_rows:
            return summary

        dice_fg_vals = [r["mean_dice_fg"] for r in self.per_image_rows if r["mean_dice_fg"] is not None]
        nsd_fg_vals  = [r["mean_nsd_fg"]  for r in self.per_image_rows if r["mean_nsd_fg"]  is not None]
        hd95_fg_vals = [r["mean_hd95_fg"] for r in self.per_image_rows if r["mean_hd95_fg"] is not None]

        summary["macro_mean_dice_fg"] = float(np.mean(dice_fg_vals)) if dice_fg_vals else None
        summary["macro_mean_nsd_fg"]  = float(np.mean(nsd_fg_vals))  if nsd_fg_vals  else None
        summary["macro_mean_hd95_fg"] = float(np.mean(hd95_fg_vals)) if hd95_fg_vals else None
        summary["fail_rate"] = float(np.mean([
            float(r["mean_dice_fg"] < 0.10)
            for r in self.per_image_rows if r["mean_dice_fg"] is not None
        ]))

        # per-class mean (TN excluded)
        summary["class_dice_macro"] = [safe_mean(v) for v in self.class_dice_values]
        summary["class_nsd"]        = [safe_mean(v) for v in self.class_nsd_values]
        summary["class_hd95"]       = [safe_mean(v) for v in self.class_hd95_values]

        lo, hi = bootstrap_ci(dice_fg_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)
        summary["macro_mean_dice_fg_ci_lower"] = lo
        summary["macro_mean_dice_fg_ci_upper"] = hi

        lo, hi = bootstrap_ci(nsd_fg_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)
        summary["macro_mean_nsd_fg_ci_lower"] = lo
        summary["macro_mean_nsd_fg_ci_upper"] = hi

        lo, hi = bootstrap_ci(hd95_fg_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)
        summary["macro_mean_hd95_fg_ci_lower"] = lo
        summary["macro_mean_hd95_fg_ci_upper"] = hi

        return summary

    def compute_per_class_summary(self) -> List[Dict]:
        rows = []
        
        # Extract per-class sensitivity/specificity from the confusion matrix
        conf = self.confmat.float()
        tp = torch.diag(conf)
        fp = conf.sum(dim=0) - tp
        fn = conf.sum(dim=1) - tp
        tn = conf.sum() - (tp + fp + fn)
        eps = 1e-6
        sensitivity = ((tp + eps) / (tp + fn + eps)).cpu().tolist()
        specificity  = ((tn + eps) / (tn + fp + eps)).cpu().tolist()
        
        for cls_idx, cls_name in enumerate(self.class_names):
            dice_vals = self.class_dice_values[cls_idx]
            nsd_vals  = self.class_nsd_values[cls_idx]
            hd95_vals = self.class_hd95_values[cls_idx]

            dice_lo, dice_hi = bootstrap_ci(dice_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)
            nsd_lo,  nsd_hi  = bootstrap_ci(nsd_vals,  n_bootstrap=self.n_bootstrap, ci=self.ci)
            hd95_lo, hd95_hi = bootstrap_ci(hd95_vals, n_bootstrap=self.n_bootstrap, ci=self.ci)

            rows.append({
                "class_idx":      cls_idx,
                "class_name":     cls_name,
                "n_evaluated":    len(dice_vals),
                "dice_mean":      safe_mean(dice_vals),
                "dice_std":       safe_std(dice_vals),
                "dice_ci_lower":  dice_lo,
                "dice_ci_upper":  dice_hi,
                "sensitivity":    sensitivity[cls_idx],   # ← added
                "specificity":    specificity[cls_idx],   # ← added
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
# Console output
# ---------------------------------------------------------------

def print_summary(per_class_summary, summary, split, ci):
    print(f"\n{'='*110}")
    print(f"Results [{split}]  n={summary.get('num_evaluated', '?')}  ({ci}% CI, bootstrap)")
    print(f"{'='*110}")
    print(f"{'Class':<30} {'n':>5} {'DSC':>6}  {'95% CI':<20} {'Sens':>6}  {'Spec':>6}  {'HD95':>7}  {'95% CI'}")
    print(f"{'-'*110}")
    for row in per_class_summary:
        if row["class_idx"] == 0:
            continue
        dice    = f"{row['dice_mean']:.3f}"  if row["dice_mean"]  is not None else "  N/A"
        sens    = f"{row['sensitivity']:.3f}" if row["sensitivity"] is not None else "  N/A"
        spec    = f"{row['specificity']:.3f}" if row["specificity"] is not None else "  N/A"
        hd95    = f"{row['hd95_mean']:.1f}"  if row["hd95_mean"]  is not None else "   N/A"
        dice_ci = (f"[{row['dice_ci_lower']:.3f}, {row['dice_ci_upper']:.3f}]"
                   if row["dice_ci_lower"] is not None else "N/A")
        hd95_ci = (f"[{row['hd95_ci_lower']:.1f}, {row['hd95_ci_upper']:.1f}]"
                   if row["hd95_ci_lower"] is not None else "N/A")
        print(f"{row['class_name']:<30} {row['n_evaluated']:>5} {dice:>6}  {dice_ci:<20} {sens:>6}  {spec:>6}  {hd95:>7}  {hd95_ci}")

    print(f"{'-'*110}")
    mean_dice = summary.get("macro_mean_dice_fg") or 0
    mean_hd95 = summary.get("macro_mean_hd95_fg") or 0
    dice_lo   = summary.get("macro_mean_dice_fg_ci_lower")
    dice_hi   = summary.get("macro_mean_dice_fg_ci_upper")
    hd95_lo   = summary.get("macro_mean_hd95_fg_ci_lower")
    hd95_hi   = summary.get("macro_mean_hd95_fg_ci_upper")
    dice_ci = f"[{dice_lo:.3f}, {dice_hi:.3f}]" if dice_lo is not None else "N/A"
    hd95_ci = f"[{hd95_lo:.1f}, {hd95_hi:.1f}]" if hd95_lo is not None else "N/A"
    n_total = summary.get("num_evaluated", "?")

    # Mean row: sens/spec are averaged over fg classes
    fg_rows = [r for r in per_class_summary if r["class_idx"] > 0]
    mean_sens = np.mean([r["sensitivity"] for r in fg_rows]) if fg_rows else 0
    mean_spec = np.mean([r["specificity"] for r in fg_rows]) if fg_rows else 0
    print(f"{'Mean (excl. bg)':<30} {n_total:>5} {mean_dice:>6.3f}  {dice_ci:<20} {mean_sens:>6.3f}  {mean_spec:>6.3f}  {mean_hd95:>7.1f}  {hd95_ci}")
    print(f"Fail rate (Dice<0.1): {summary.get('fail_rate', 0)*100:.1f}%")
    print(f"{'='*110}")


# ---------------------------------------------------------------
# Save CSV / JSON
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
    parser.add_argument("--gt_dir",     default="./TEP_dataset/ext_val/masks",  help="GT mask folder (index PNG)")
    parser.add_argument("--pred_dir",   default=".//ext_val",  help="Pred mask folder (index PNG)")
    parser.add_argument("--output_dir", default="./final_results")
    parser.add_argument("--split",      default="ext_val", help="For result labeling (e.g., int_val / int_val)")
    parser.add_argument("--n_bootstrap", type=int,   default=1000)
    parser.add_argument("--ci",          type=float, default=95.0)
    args = parser.parse_args()

    gt_dir   = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    save_dir = Path(args.output_dir)

    gt_paths = sorted(gt_dir.glob("*.png"))
    if not gt_paths:
        gt_paths = sorted(gt_dir.glob("*.PNG"))
    print(f"[{args.split}] GT: {len(gt_paths)} images")

    agg = MetricsAggregator(
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        nsd_tau=NSD_TAU,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
    )

    missing_preds = []
    for gt_path in tqdm(gt_paths, desc=f"Evaluating [{args.split}]"):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            missing_preds.append(gt_path.name)
            continue

        gt_mask   = np.array(Image.open(gt_path)).astype(np.int64)
        pred_mask = np.array(Image.open(pred_path)).astype(np.int64)

        agg.update(gt_mask, pred_mask, image_name=gt_path.name)

    if missing_preds:
        print(f"[WARN] {len(missing_preds)} pred masks not found: {missing_preds[:5]}{'...' if len(missing_preds) > 5 else ''}")

    print(f"\nComputing bootstrap CI (n={args.n_bootstrap})...")
    summary = {
        "split":        args.split,
        "gt_dir":       str(gt_dir),
        "pred_dir":     str(pred_dir),
        "num_evaluated": len(agg.per_image_rows),
        "nsd_tau":      NSD_TAU,
        "n_bootstrap":  args.n_bootstrap,
        "ci":           args.ci,
        **agg.compute_summary(),
    }
    per_class_summary = agg.compute_per_class_summary()

    write_json(summary,               save_dir / "summary.json")
    write_csv(agg.per_image_rows,     save_dir / "per_image_metrics.csv")
    write_csv(agg.class_metrics_rows, save_dir / "class_metrics.csv")
    write_csv(per_class_summary,      save_dir / "per_class_summary.csv")

    print_summary(per_class_summary, summary, args.split, args.ci)
    print(f"\nSaved → {save_dir}")


if __name__ == "__main__":
    main()
