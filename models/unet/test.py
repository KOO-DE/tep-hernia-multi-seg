"""
SMP U-Net TEP Segmentation - Test & Evaluation Script

사용법:
  python test.py --data_root data --checkpoint 260316_ro/best_dice.pth
  python test.py --data_root data --checkpoint 260316_ro/best_dice.pth --save_preds
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from tqdm import tqdm

import segmentation_models_pytorch as smp
from medpy.metric.binary import dc, precision, recall
from scipy.ndimage import distance_transform_edt

# train.py에서 필요한 것들 import
from train import (
    SegDataset,
    dice_score,
    pixel_accuracy,
)


# ============================================================
# 클래스 정보
# ============================================================

CLASS_NAMES = ["Background", "Inf. Epigastric Vv.", "Pubic Arch", "Spermatic Cord", "Vas Deferens"]

CLASS_COLORS = [
    [0,   0,   0  ],   # 0: Background
    [231, 226, 203],   # 1: Inf. Epigastric Vv.
    [252, 155, 192],   # 2: Pubic Arch
    [176, 145, 131],   # 3: Spermatic Cord
    [169, 117, 87 ],   # 4: Vas Deferens
]


class SegDatasetWithNames(SegDataset):
    """파일명도 함께 반환하는 Dataset (test 전용)"""
    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)
        return image, mask, self.ids[idx]   # fname 추가


def decode_mask(mask_tensor):
    """(H, W) int64 -> (H, W, 3) RGB"""
    mask = mask_tensor.cpu().numpy()
    rgb  = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, color in enumerate(CLASS_COLORS):
        rgb[mask == c] = color
    return rgb


# ============================================================
# Args
# ============================================================

def get_args():
    parser = argparse.ArgumentParser(description="TEP SMP U-Net Test")
    parser.add_argument("--data_root",   type=str, default="TEP_dataset/ro",                  help="데이터 루트 경로")
    parser.add_argument("--checkpoint",  type=str, default="260401_ro/best_dice.pth", help="체크포인트 경로")
    parser.add_argument("--encoder",     type=str, default="resnet50",               help="train.py와 동일한 encoder")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--img_size",    type=int, default=512)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir",  type=str, default="predictions/260401_ro/dice/int_val",           help="결과 저장 폴더")
    parser.add_argument("--save_preds",  action="store_true",                        help="샘플별 예측 이미지 저장")
    parser.add_argument("--n_vis",       type=int, default=8,                        help="격자 시각화 샘플 수")
    return parser.parse_args()


# ============================================================
# 클래스별 Dice / IoU 계산
# ============================================================

def compute_nsd(pred_mask, gt_mask, tolerance=2):
    """
    Surface Distance 기반 NSD (0~1, 높을수록 좋음)
    tolerance: 허용 거리 (픽셀 단위, 기본값 2)
    """
    if gt_mask.sum() == 0 and pred_mask.sum() == 0:
        return 1.0
    if gt_mask.sum() == 0 or pred_mask.sum() == 0:
        return 0.0

    # surface 추출: 마스크 내부에서 경계 픽셀만
    def get_surface(mask):
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        return mask & ~eroded

    pred_surface = get_surface(pred_mask)
    gt_surface   = get_surface(gt_mask)

    # 각 surface에서 상대방까지의 거리 변환
    dist_pred_to_gt = distance_transform_edt(~gt_mask)
    dist_gt_to_pred = distance_transform_edt(~pred_mask)

    # tolerance 이내인 surface 픽셀 비율
    pred_within = (dist_pred_to_gt[pred_surface] <= tolerance).sum()
    gt_within   = (dist_gt_to_pred[gt_surface]   <= tolerance).sum()

    nsd = (pred_within + gt_within) / (pred_surface.sum() + gt_surface.sum() + 1e-6)
    return float(np.clip(nsd, 0.0, 1.0))


def compute_per_class_metrics(pred, target, num_classes, smooth=1e-6):
    """
    pred, target: (B, H, W) int64
    Dice, IoU, Precision, Recall, NSD(hd95) 계산
    background(0) 제외한 mean 반환
    """
    pred_one_hot   = torch.zeros(pred.shape[0], num_classes, *pred.shape[1:]).to(pred.device)
    target_one_hot = torch.zeros_like(pred_one_hot)
    pred_one_hot.scatter_(1, pred.unsqueeze(1), 1)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)

    result = {}
    dice_list      = []
    iou_list       = []
    precision_list = []
    recall_list    = []
    nsd_list       = []

    for c in range(num_classes):
        p = pred_one_hot[:, c].float()
        t = target_one_hot[:, c].float()

        intersection = (p * t).sum()
        union        = p.sum() + t.sum() - intersection

        dice_val = (2 * intersection + smooth) / (p.sum() + t.sum() + smooth)
        iou_val  = (intersection + smooth) / (union + smooth)
        prec_val = (intersection + smooth) / (p.sum() + smooth)
        rec_val  = (intersection + smooth) / (t.sum() + smooth)

        # NSD: surface distance 기반 (0~1, tolerance=2px)
        p_np    = p.cpu().numpy().astype(bool)
        t_np    = t.cpu().numpy().astype(bool)
        nsd_val = compute_nsd(p_np, t_np, tolerance=2)

        result[f"dice_{CLASS_NAMES[c]}"]      = dice_val.item()
        result[f"iou_{CLASS_NAMES[c]}"]       = iou_val.item()
        result[f"precision_{CLASS_NAMES[c]}"] = prec_val.item()
        result[f"recall_{CLASS_NAMES[c]}"]    = rec_val.item()
        result[f"nsd_{CLASS_NAMES[c]}"]       = nsd_val

        dice_list.append(dice_val.item())
        iou_list.append(iou_val.item())
        precision_list.append(prec_val.item())
        recall_list.append(rec_val.item())
        nsd_list.append(nsd_val)

    # mean: background 제외
    result["mean_dice"]      = float(np.mean(dice_list[1:]))
    result["mean_iou"]       = float(np.mean(iou_list[1:]))
    result["mean_precision"] = float(np.mean(precision_list[1:]))
    result["mean_recall"]    = float(np.mean(recall_list[1:]))
    result["mean_nsd"]       = float(np.mean(nsd_list[1:]))

    return result


# ============================================================
# 평가
# ============================================================

@torch.no_grad()
def run_test(model, loader, num_classes, device):
    model.eval()
    all_metrics = {}
    all_preds   = []
    all_masks   = []
    all_images  = []
    all_fnames  = []

    for images, masks, fnames in tqdm(loader, desc="Testing"):
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)          # (B, C, H, W)
        preds  = logits.argmax(dim=1)   # (B, H, W)

        m = compute_per_class_metrics(preds, masks, num_classes)
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)

        all_preds.append(preds.cpu())
        all_masks.append(masks.cpu())
        all_images.append(images.cpu())
        all_fnames.extend(fnames)

    avg        = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    all_preds  = torch.cat(all_preds,  dim=0)
    all_masks  = torch.cat(all_masks,  dim=0)
    all_images = torch.cat(all_images, dim=0)

    return avg, all_images, all_masks, all_preds, all_fnames


# ============================================================
# 결과 출력 및 저장
# ============================================================

def print_results(metrics):
    print("\n" + "=" * 55)
    print("  Test Results")
    print("=" * 55)
    print(f"  {'Mean Dice (no BG)':<28} {metrics['mean_dice']:.4f}")
    print(f"  {'Mean IoU  (no BG)':<28} {metrics['mean_iou']:.4f}")
    print("-" * 55)
    for name in CLASS_NAMES[1:]:
        print(f"  {name:<28} "
              f"Dice: {metrics[f'dice_{name}']:.4f}  "
              f"IoU:  {metrics[f'iou_{name}']:.4f}")
    print("=" * 55)


def save_metrics_csv(metrics, output_dir):
    rows = []
    for name in CLASS_NAMES[1:]:
        rows.append({
            "class":     name,
            "dice":      metrics[f"dice_{name}"],
            "iou":       metrics[f"iou_{name}"],
            "precision": metrics[f"precision_{name}"],
            "recall":    metrics[f"recall_{name}"],
            "nsd":       metrics[f"nsd_{name}"],
        })
    rows.append({
        "class":     "Mean (no BG)",
        "dice":      metrics["mean_dice"],
        "iou":       metrics["mean_iou"],
        "precision": metrics["mean_precision"],
        "recall":    metrics["mean_recall"],
        "nsd":       metrics["mean_nsd"],
    })
    path = os.path.join(output_dir, "test_metrics.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Metrics saved      -> {path}")


# ============================================================
# 시각화
# ============================================================

def denormalize(tensor):
    """(3, H, W) tensor -> (H, W, 3) numpy 0~1"""
    img = tensor.permute(1, 2, 0).numpy()
    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)


def save_pred_masks(preds, fnames, output_dir):
    """예측 마스크만 RGB PNG로 저장"""
    mask_dir = os.path.join(output_dir, "pred_masks")
    os.makedirs(mask_dir, exist_ok=True)

    for i in tqdm(range(len(preds)), desc="Saving pred masks"):
        pred_rgb  = decode_mask(preds[i])   # (H, W, 3) uint8
        save_name = fnames[i]
        if not save_name.endswith(".png"):
            save_name = save_name + ".png"
        plt.imsave(os.path.join(mask_dir, save_name), pred_rgb)

    print(f"  Pred masks saved   -> {mask_dir}/")


def save_individual_preds(images, masks, preds, fnames, output_dir):
    pred_dir = os.path.join(output_dir, "individual")
    os.makedirs(pred_dir, exist_ok=True)

    for i in tqdm(range(len(images)), desc="Saving individual"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        img      = denormalize(images[i])
        gt_rgb   = decode_mask(masks[i])
        pred_rgb = decode_mask(preds[i])

        axes[0].imshow(img);       axes[0].set_title("Image")
        axes[1].imshow(gt_rgb);    axes[1].set_title("GT Mask")
        axes[2].imshow(pred_rgb);  axes[2].set_title("Prediction")
        for ax in axes:
            ax.axis("off")

        patches = [
            mpatches.Patch(color=np.array(c) / 255, label=name)
            for c, name in zip(CLASS_COLORS[1:], CLASS_NAMES[1:])
        ]
        fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        # 원본 파일명으로 저장
        save_name = fnames[i] + ".png" if not fnames[i].endswith(".png") else fnames[i]
        plt.savefig(os.path.join(pred_dir, save_name), dpi=100)
        plt.close()

    print(f"  Individual preds   -> {pred_dir}/")


def save_class_bar(metrics, output_dir):
    names = CLASS_NAMES[1:]
    dice  = [metrics[f"dice_{n}"] for n in names]
    iou   = [metrics[f"iou_{n}"]  for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - w/2, dice, w, label="Dice", color="steelblue")
    ax.bar(x + w/2, iou,  w, label="IoU",  color="coral")
    ax.axhline(metrics["mean_dice"], color="steelblue", linestyle="--", alpha=0.7,
               label=f"Mean Dice {metrics['mean_dice']:.3f}")
    ax.axhline(metrics["mean_iou"],  color="coral",     linestyle="--", alpha=0.7,
               label=f"Mean IoU  {metrics['mean_iou']:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Dice & IoU")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "class_metrics_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Bar chart saved    -> {path}")


# ============================================================
# Main
# ============================================================

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 모델 로드 - train.py와 동일한 encoder 사용
    model = smp.Unet(
        encoder_name    = args.encoder,
        encoder_weights = None,          # 평가 시에는 pretrained 불필요
        in_channels     = 3,
        classes         = args.num_classes,
    ).to(device)

    # checkpoint 로드 (train.py의 save_checkpoint 구조에 맞춤)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Checkpoint loaded: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")

    # DataLoader
    test_ds = SegDatasetWithNames(
        data_root=os.path.join(args.data_root, "int_val"),
        img_size=args.img_size,
        augment=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"Test samples: {len(test_ds)}")

    # 평가
    metrics, images, masks, preds, fnames = run_test(model, test_loader, args.num_classes, device)

    # 결과 출력 및 저장
    print_results(metrics)
    save_metrics_csv(metrics, args.output_dir)
    save_class_bar(metrics, args.output_dir)
    save_pred_masks(preds, fnames, args.output_dir)   # pred masks 저장 (output_dir/pred_masks/)

    if args.save_preds:
        save_individual_preds(images, masks, preds, fnames, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
