# -*- coding: utf-8 -*-
"""
inference_uncertainty.py


Usage:
    python inference_uncertainty.py \
        --config configs/dinov2/tep/semantic/eomt_large_504.yaml \
        --ckpt logs/tep_dinov2_vitl/ori_aug/checkpoints/best.ckpt \
        --per_image_csv results/ori_aug/int_val/per_image_metrics.csv \
        --class_metrics_csv results/ori_aug/int_val/class_metrics.csv \
        --data_path TEP_dataset/ro \
        --split int_val \
        --output_dir results/uncertainty/ori_aug \
        --mc_passes 100 \
        --bad_thresholds 0.5 \
        --good_thresholds 0.8 \
        --dropout_p 0.1
"""

import argparse
import csv
import importlib
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from lightning import seed_everything
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
CLS_NAMES_SHORT = {1: "iev", 2: "pubic", 3: "spermatic", 4: "vas"}

CLASS_TO_COLOR = {
    0: (0, 0, 0),
    1: (0, 255, 255),      # IEV: 청록
    2: (0, 255, 0),        # Pubic arch: 초록
    3: (147, 112, 219),    # Spermatic cord: 보라
    4: (0, 0, 255),        # Vas deferens: 파랑
}


# ---------------------------------------------------------------
# 시각화 유틸
# ---------------------------------------------------------------

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls, color in CLASS_TO_COLOR.items():
        out[mask == cls] = np.asarray(color, dtype=np.uint8)
    return out


def overlay_mask_on_image(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    for cls, color in CLASS_TO_COLOR.items():
        if cls == 0:
            continue
        fg = mask == cls
        if fg.any():
            color_arr = np.array(color, dtype=np.float32)
            out[fg] = (1.0 - alpha) * out[fg] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def get_valid_mask(img_np: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    복강경 영상의 유효 영역 마스크 추출.
    어두운 가장자리(vignetting) 영역을 제거.
    반환: (H, W) bool mask, True=유효 영역
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def uncertainty_to_heatmap(uncertainty: np.ndarray, colormap=cv2.COLORMAP_JET,
                           global_max: Optional[float] = None,
                           with_colorbar: bool = False,
                           vmax_label: Optional[float] = None) -> np.ndarray:
    """
    Per-pixel uncertainty → heatmap PNG.
    1단계: 이미지별 99th percentile로 outlier 클리핑
    2단계: global_max가 있으면 fixed max로 정규화, 없으면 이미지별 정규화
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    # 1단계: 이미지별 99th percentile로 outlier 클리핑
    p99 = float(np.percentile(uncertainty, 99))
    blurred = cv2.GaussianBlur(
        np.clip(uncertainty, 0, p99).astype(np.float32), (21, 21), 0)

    # 2단계: 정규화
    if global_max is not None:
        e_min = 0.0
        e_max = float(global_max)
    else:
        e_min = float(np.percentile(blurred, 5))
        e_max = float(np.percentile(blurred, 95))

    if e_max - e_min < 1e-8:
        normalized = np.zeros_like(blurred)
    else:
        normalized = np.clip((blurred - e_min) / (e_max - e_min), 0, 1)

    if not with_colorbar:
        gray = (normalized * 255).astype(np.uint8)
        return cv2.applyColorMap(gray, colormap)

    # matplotlib으로 colorbar 포함 heatmap 생성
    h, w = blurred.shape[:2]
    fig_w = w / 100.0
    fig_h = h / 100.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    vmax = vmax_label if vmax_label is not None else (e_max if global_max is None else global_max)
    im = ax.imshow(blurred, cmap="jet", vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Uncertainty", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.axis("off")
    plt.tight_layout(pad=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)

    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img_rgb = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)  # RGB 반환


# ---------------------------------------------------------------
# CSV에서 Dice 읽기
# ---------------------------------------------------------------

def load_per_image_dice(csv_path: str) -> Dict[str, float]:
    image_dice = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_dice[row["image_name"]] = float(row["mean_dice_fg"])
    print(f"CSV 로드: {len(image_dice)}장 ({csv_path})")
    return image_dice


def load_per_class_dice(csv_path: str) -> Dict[str, Dict[int, float]]:
    from collections import defaultdict
    image_class_dice = defaultdict(dict)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_idx = int(row["class_idx"])
            if cls_idx == 0:
                continue
            image_class_dice[row["image_name"]][cls_idx] = float(row["dice"])
    print(f"Class metrics CSV 로드: {len(image_class_dice)}장 ({csv_path})")
    return dict(image_class_dice)


def select_cases_by_class_dice(
    image_class_dice: Dict[str, Dict[int, float]],
    mode: str,
    threshold: float,
    gt_mask_dir: Path,
    target_class: Optional[int] = None,
) -> Dict[str, float]:
    """
    mode="good": Dice > threshold
    mode="bad":  Dice < threshold
    target_class: 특정 클래스 인덱스 지정 시 해당 클래스만 기준으로 선별
                  None이면 모든 fg 클래스 기준 (기존 동작)
    """
    selected = {}
    for img_name, class_dice in image_class_dice.items():
        if not class_dice:
            continue
        # 4개 fg 클래스가 모두 있는 케이스만 선별 (dice > 0인 클래스 기준)
        if sum(1 for d in class_dice.values() if d > 0) < 4:
            continue
        if target_class is not None:
            if target_class not in class_dice:
                continue
            dice_vals = [class_dice[target_class]]
        else:
            dice_vals = list(class_dice.values())
        mean_dice = float(np.mean(list(class_dice.values())))
        if mode == "good":
            if all(d > threshold for d in dice_vals):
                selected[img_name] = mean_dice
        elif mode == "bad":
            if all(d < threshold for d in dice_vals):
                selected[img_name] = mean_dice
    return selected


# ---------------------------------------------------------------
# EoMT model 로드
# ---------------------------------------------------------------

def load_model(config_path: str, ckpt_path: str, device: str, dropout_p: float = 0.1):
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
    net_kwargs["mc_dropout_p"] = dropout_p
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
# MC Dropout inference
# ---------------------------------------------------------------

@torch.no_grad()
def infer_mc_dropout(
    model, img_np: np.ndarray, img_size: tuple, device: str, n_passes: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MC Dropout 기반 Prediction Variance uncertainty.
    - n_passes번 forward pass → argmax 예측 불일치 비율로 uncertainty 계산
    - fg 픽셀 기반 uncertainty_mean 계산
 
    반환:
        mean_pred:         (H, W) 최빈 클래스 예측
        uncertainty:       (H, W) 예측 불일치 비율 [0, 1]
        class_uncertainty: (NUM_CLASSES, H, W) 클래스별 uncertainty
    """
    orig_h, orig_w = img_np.shape[:2]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    img_resized = F.interpolate(
        img_tensor.unsqueeze(0), size=img_size, mode="bilinear", align_corners=False
    )[0]
    img_batch = img_resized.unsqueeze(0).to(device)
 
    all_preds = []
 
    for _ in range(n_passes):
        with torch.cuda.amp.autocast():
            img_batch_norm = img_batch / 255.0
            mask_logits_list, class_logits_list = model.network(img_batch_norm)
 
        mask_logits = F.interpolate(mask_logits_list[-1], img_size, mode="bilinear")
        logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits_list[-1])
        logits_resized = F.interpolate(
            logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        pred = logits_resized[0].argmax(dim=0).cpu().numpy().astype(np.int32)
        all_preds.append(pred)
 
    all_preds = np.stack(all_preds, axis=0)  # [N, H, W]
 
    # 전체 uncertainty: 최빈값 아닌 예측 비율
    from scipy.stats import mode as scipy_mode
    mode_result = scipy_mode(all_preds, axis=0)
    mean_pred = mode_result.mode.astype(np.int64)
    if mean_pred.ndim == 3:
        mean_pred = mean_pred[0]
 
    mode_count = mode_result.count
    if mode_count.ndim == 3:
        mode_count = mode_count[0]
    uncertainty = 1.0 - (mode_count.astype(np.float32) / n_passes)
 
    # 클래스별 uncertainty
    class_uncertainty = np.zeros((NUM_CLASSES, orig_h, orig_w), dtype=np.float32)
    for cls_idx in range(NUM_CLASSES):
        cls_freq = (all_preds == cls_idx).mean(axis=0).astype(np.float32)
        class_uncertainty[cls_idx] = 1.0 - np.abs(cls_freq * 2 - 1)
 
    # 가장자리 마스킹
    valid_mask = get_valid_mask(img_np)
    uncertainty[~valid_mask] = 0.0
    for cls_idx in range(NUM_CLASSES):
        class_uncertainty[cls_idx][~valid_mask] = 0.0
 
    return mean_pred, uncertainty, class_uncertainty


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--per_image_csv", required=True)
    parser.add_argument("--class_metrics_csv", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--split", default="int_val",
                        choices=["int_val", "ext_val", "train", "tune"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", default="results/uncertainty")
    parser.add_argument("--mc_passes", type=int, default=100)
    parser.add_argument("--bad_thresholds", type=float, nargs="+", default=[0.3])
    parser.add_argument("--good_thresholds", type=float, nargs="+", default=[0.8])
    parser.add_argument("--uncertainty_max", type=float, default=None,
                        help="uncertainty heatmap 정규화 기준값 고정 (예: 0.07). "
                             "없으면 전체 95th percentile 자동 계산")
    parser.add_argument("--orig_size", type=int, nargs=2, default=None,
                        metavar=("W", "H"),
                        help="저장 시 resize할 원본 크기 (W H), 예: --orig_size 1024 526")
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--target_class", type=int, default=None,
                        help="특정 클래스만 기준으로 케이스 선별 (예: 4=vas_deferens). None이면 모든 fg 클래스 기준")
    parser.add_argument("--all_images", action="store_true",
                        help="Dice 기준 없이 모든 이미지에 대해 uncertainty map 저장")
    args = parser.parse_args()

    device = f"cuda:{args.device}"

    if args.data_path is None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        args.data_path = cfg["data"]["init_args"]["path"]

    image_dice = load_per_image_dice(args.per_image_csv)
    mask_dir = Path(args.data_path) / args.split / "masks"

    if args.all_images:
        print("\n▶ 전체 이미지 대상 (Dice 기준 없음)")
        all_target_images = set(image_dice.keys())
        print(f"  전체 이미지: {len(all_target_images)}장")
    elif args.class_metrics_csv:
        print("\n▶ Per-class Dice 기반 케이스 선별")
        image_class_dice = load_per_class_dice(args.class_metrics_csv)
        all_target_images = set()
        for thr in args.bad_thresholds:
            cases = select_cases_by_class_dice(image_class_dice, "bad", thr, mask_dir, args.target_class)
            print(f"  Bad (모든 fg 클래스 Dice < {thr:.1f}): {len(cases)}장")
            all_target_images.update(cases.keys())
        for thr in args.good_thresholds:
            cases = select_cases_by_class_dice(image_class_dice, "good", thr, mask_dir, args.target_class)
            print(f"  Good (모든 fg 클래스 Dice > {thr:.1f}): {len(cases)}장")
            all_target_images.update(cases.keys())
    else:
        print("\n▶ Mean Dice 기반 케이스 선별")
        dice_values = list(image_dice.values())
        print(f"Dice 분포: mean={np.mean(dice_values):.3f}, "
              f"min={np.min(dice_values):.3f}, max={np.max(dice_values):.3f}")
        all_target_images = set()
        for thr in args.bad_thresholds:
            n = sum(1 for d in dice_values if d < thr)
            print(f"  Bad Dice < {thr:.1f}: {n}장")
            all_target_images.update(name for name, d in image_dice.items() if d < thr)
        for thr in args.good_thresholds:
            n = sum(1 for d in dice_values if d > thr)
            print(f"  Good Dice > {thr:.1f}: {n}장")
            all_target_images.update(name for name, d in image_dice.items() if d > thr)

    print(f"\n  MC Dropout 적용 대상: {len(all_target_images)}장")

    if len(all_target_images) == 0:
        print("해당 케이스 없음.")
        return

    model, img_size = load_model(args.config, args.ckpt, device, dropout_p=args.dropout_p)

    img_dir = Path(args.data_path) / args.split / "images"
    mask_dir = Path(args.data_path) / args.split / "masks"
    save_dir = Path(args.output_dir)

    print(f"\n▶ MC Dropout inference ({args.mc_passes}회, prediction variance)")
    mc_results = {}

    for img_name in tqdm(sorted(all_target_images), desc="MC Dropout"):
        img_path = img_dir / img_name
        mask_path = mask_dir / img_name
        if not img_path.exists() or not mask_path.exists():
            print(f"[WARN] 파일 없음: {img_name}")
            continue

        img_np = np.array(Image.open(img_path).convert("RGB"))
        gt_mask = np.array(Image.open(mask_path)).astype(np.int64)

        mean_pred, uncertainty, class_uncertainty = infer_mc_dropout(
            model, img_np, img_size, device, n_passes=args.mc_passes
        )
        mc_results[img_name] = {
            "img_np": img_np,
            "gt_mask": gt_mask,
            "mean_pred": mean_pred,
            "uncertainty": uncertainty,
            "class_uncertainty": class_uncertainty,
            "dice": image_dice[img_name],
        }

    # global_max 계산
    all_uncertainties = np.concatenate([
        d["uncertainty"].flatten() for d in mc_results.values()
    ])
    if args.uncertainty_max is not None:
        p95_global = float(np.percentile(all_uncertainties[all_uncertainties > 0], 95))
        global_max = min(p95_global, float(args.uncertainty_max))
        print(f"\n  Global uncertainty max (95th={p95_global:.4f}, fixed cap={args.uncertainty_max:.4f}): {global_max:.4f}")
    else:
        global_max = float(np.percentile(all_uncertainties[all_uncertainties > 0], 95))
        print(f"\n  Global uncertainty max (95th percentile): {global_max:.4f}")

    print("\n▶ 저장 중...")
    summary_all = {}

    def save_cases(target_cases, group_dir, label, img_dir, orig_size=None):
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "original").mkdir(parents=True, exist_ok=True)
        (group_dir / "gt_mask").mkdir(parents=True, exist_ok=True)
        (group_dir / "uncertainty_heatmap").mkdir(parents=True, exist_ok=True)
        (group_dir / "uncertainty_heatmap_global").mkdir(parents=True, exist_ok=True)
        (group_dir / "overlay_pred").mkdir(parents=True, exist_ok=True)
        (group_dir / "overlay_gt").mkdir(parents=True, exist_ok=True)
        (group_dir / "uncertainty_overlay").mkdir(parents=True, exist_ok=True)
        (group_dir / "uncertainty_overlay_global").mkdir(parents=True, exist_ok=True)
        for cls_idx, cls_short in CLS_NAMES_SHORT.items():
            (group_dir / f"uncertainty_class{cls_idx}_{cls_short}").mkdir(parents=True, exist_ok=True)
            (group_dir / f"uncertainty_class{cls_idx}_{cls_short}_global").mkdir(parents=True, exist_ok=True)
        (group_dir / "uncertainty_heatmap_global_colorbar").mkdir(parents=True, exist_ok=True)

        print(f"\n  [{label}] {len(target_cases)}장")
        case_summary = []

        for img_name, data in sorted(target_cases.items(), key=lambda x: x[1]["dice"]):
            img_np = data["img_np"]
            gt_mask = data["gt_mask"]
            mean_pred = data["mean_pred"]
            uncertainty = data["uncertainty"]
            class_uncertainty = data["class_uncertainty"]
            dice = data["dice"]

            # orig_size로 resize (있을 때만)
            if orig_size is not None:
                orig_w, orig_h = orig_size
                img_np = cv2.resize(img_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                mean_pred = cv2.resize(
                    mean_pred.astype(np.uint8), (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST).astype(np.int64)
                gt_mask = cv2.resize(
                    gt_mask.astype(np.uint8), (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST).astype(np.int64)
                uncertainty = cv2.resize(
                    uncertainty, (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR)
                class_uncertainty = np.stack([
                    cv2.resize(class_uncertainty[i], (orig_w, orig_h),
                               interpolation=cv2.INTER_LINEAR)
                    for i in range(class_uncertainty.shape[0])
                ], axis=0)

            Image.fromarray(img_np).save(group_dir / "original" / img_name)
            Image.fromarray(colorize_mask(gt_mask.astype(np.int64))).save(
                group_dir / "gt_mask" / img_name)

            # 전체 uncertainty (이미지별)
            heatmap_bgr = uncertainty_to_heatmap(uncertainty)
            cv2.imwrite(str(group_dir / "uncertainty_heatmap" / img_name), heatmap_bgr)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(cv2.addWeighted(img_np, 0.7, heatmap_rgb, 0.3, 0)).save(
                group_dir / "uncertainty_overlay" / img_name)

            # 전체 uncertainty (global)
            heatmap_bgr_g = uncertainty_to_heatmap(uncertainty, global_max=global_max)
            cv2.imwrite(str(group_dir / "uncertainty_heatmap_global" / img_name), heatmap_bgr_g)
            heatmap_rgb_g = cv2.cvtColor(heatmap_bgr_g, cv2.COLOR_BGR2RGB)
            Image.fromarray(cv2.addWeighted(img_np, 0.7, heatmap_rgb_g, 0.3, 0)).save(
                group_dir / "uncertainty_overlay_global" / img_name)

            # 전체 uncertainty (global + colorbar)
            heatmap_cb = uncertainty_to_heatmap(
                uncertainty, global_max=global_max,
                with_colorbar=True, vmax_label=global_max
            )
            Image.fromarray(heatmap_cb).save(
                group_dir / "uncertainty_heatmap_global_colorbar" / img_name)

            # 클래스별 uncertainty
            for cls_idx, cls_short in CLS_NAMES_SHORT.items():
                cls_heatmap = uncertainty_to_heatmap(class_uncertainty[cls_idx])
                cv2.imwrite(
                    str(group_dir / f"uncertainty_class{cls_idx}_{cls_short}" / img_name),
                    cls_heatmap)
                cls_heatmap_g = uncertainty_to_heatmap(class_uncertainty[cls_idx], global_max=global_max)
                cv2.imwrite(
                    str(group_dir / f"uncertainty_class{cls_idx}_{cls_short}_global" / img_name),
                    cls_heatmap_g)

            Image.fromarray(overlay_mask_on_image(img_np, mean_pred)).save(
                group_dir / "overlay_pred" / img_name)
            Image.fromarray(overlay_mask_on_image(img_np, gt_mask)).save(
                group_dir / "overlay_gt" / img_name)

            case_summary.append({
                "image_name": img_name,
                "dice": float(dice),
                "uncertainty_mean": float(uncertainty[mean_pred != 0].mean()) if (mean_pred != 0).any() else 0.0,
                "uncertainty_max": float(uncertainty.max()),
                "class_uncertainty_mean": {
                    CLS_NAMES_SHORT[i]: float(class_uncertainty[i].mean())
                    for i in CLS_NAMES_SHORT
                },
            })

        case_summary.sort(key=lambda x: x["dice"])
        with open(group_dir / "summary.json", "w") as f:
            json.dump({
                "label": label,
                "n_cases": len(target_cases),
                "mc_passes": args.mc_passes,
                "dropout_p": args.dropout_p,
                "uncertainty_method": "predictive_entropy_fg",
                "cases": case_summary,
            }, f, indent=2, ensure_ascii=False)

        return len(target_cases)

    if args.all_images:
        # 전체 이미지를 단일 폴더에 저장
        n = save_cases(mc_results, save_dir / "all", "all", img_dir, args.orig_size)
        summary_all["all"] = {"type": "all", "n_cases": n}
    else:
        # Bad cases
        for thr in args.bad_thresholds:
            thr_name = f"bad_dice_lt_{int(thr * 10):02d}"
            if args.class_metrics_csv:
                bad_cases = select_cases_by_class_dice(image_class_dice, "bad", thr, mask_dir, args.target_class)
                target_cases = {n: mc_results[n] for n in bad_cases if n in mc_results}
            else:
                target_cases = {n: d for n, d in mc_results.items() if d["dice"] < thr}
            n = save_cases(target_cases, save_dir / thr_name, f"Dice < {thr:.1f}", img_dir, args.orig_size)
            summary_all[thr_name] = {"type": "bad", "threshold": thr, "n_cases": n}

        # Good cases
        for thr in args.good_thresholds:
            thr_name = f"good_dice_gte_{int(thr * 10):02d}"
            if args.class_metrics_csv:
                good_cases = select_cases_by_class_dice(image_class_dice, "good", thr, mask_dir, args.target_class)
                target_cases = {n: mc_results[n] for n in good_cases if n in mc_results}
            else:
                target_cases = {n: d for n, d in mc_results.items() if d["dice"] >= thr}
            n = save_cases(target_cases, save_dir / thr_name, f"Dice >= {thr:.1f}", img_dir, args.orig_size)
            summary_all[thr_name] = {"type": "good", "threshold": thr, "n_cases": n}

    with open(save_dir / "summary_all.json", "w") as f:
        json.dump(summary_all, f, indent=2)

    print(f"\n완료! 결과 → {save_dir}")


if __name__ == "__main__":
    main()
