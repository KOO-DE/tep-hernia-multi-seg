import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import segmentation_models_pytorch as smp

from PIL import Image


# ────────────────────────────────────────────
# 1. Dataset
# ────────────────────────────────────────────
class SegDataset(Dataset):
    """
    폴더 구조:
        data_root/
            images/  *.png (or *.jpg)
            masks/   *.png  (픽셀값 = 클래스 인덱스)
    """
    def __init__(self, data_root, img_size=512, augment=False):
        self.img_dir  = os.path.join(data_root, "images")
        self.mask_dir = os.path.join(data_root, "masks")
        self.img_size = img_size
        self.augment  = augment

        self.ids = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(self.img_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ])

        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        # image
        for ext in [".png", ".jpg", ".jpeg"]:
            img_path = os.path.join(self.img_dir, name + ext)
            if os.path.exists(img_path):
                break
        image = Image.open(img_path).convert("RGB")
        image = self.img_tf(image)

        # mask
        mask_path = os.path.join(self.mask_dir, name + ".png")
        mask = Image.open(mask_path).convert("L")
        mask = self.mask_tf(mask)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask


# ────────────────────────────────────────────
# 2. Metrics helpers
# ────────────────────────────────────────────
def dice_score(pred, target, num_classes, smooth=1e-6):
    """per-class Dice, background(0) 제외 평균 반환"""
    pred_one_hot   = torch.zeros(pred.shape[0], num_classes, *pred.shape[1:]).to(pred.device)
    target_one_hot = torch.zeros_like(pred_one_hot)
    pred_one_hot.scatter_(1, pred.unsqueeze(1), 1)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)

    dice_per_class = []
    for c in range(1, num_classes):          # background 제외
        p = pred_one_hot[:, c].float()
        t = target_one_hot[:, c].float()
        intersection = (p * t).sum()
        dice = (2 * intersection + smooth) / (p.sum() + t.sum() + smooth)
        dice_per_class.append(dice.item())

    return np.mean(dice_per_class)


def pixel_accuracy(pred, target):
    """전체 픽셀 중 정답 비율"""
    correct = (pred == target).sum().item()
    total   = target.numel()
    return correct / total


# ────────────────────────────────────────────
# 3. Train one epoch
# ────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, num_classes):
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    total_dice = 0.0

    for images, masks in tqdm(loader, desc="  Train", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item()
        total_acc  += pixel_accuracy(preds, masks)
        total_dice += dice_score(preds, masks, num_classes)

    n = len(loader)
    return total_loss / n, total_acc / n, total_dice / n


# ────────────────────────────────────────────
# 4. Validate one epoch
# ────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    total_dice = 0.0

    for images, masks in tqdm(loader, desc="  Valid", leave=False):
        images, masks = images.to(device), masks.to(device)

        logits = model(images)
        loss   = criterion(logits, masks)
        preds  = logits.argmax(dim=1)

        total_loss += loss.item()
        total_acc  += pixel_accuracy(preds, masks)
        total_dice += dice_score(preds, masks, num_classes)

    n = len(loader)
    return total_loss / n, total_acc / n, total_dice / n


# ────────────────────────────────────────────
# 5. Curve PNG 저장 (매 epoch 갱신)
# ────────────────────────────────────────────
def save_curves(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    # Pixel Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Pixel Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    # Dice
    axes[2].plot(epochs, history["train_dice"], label="Train")
    axes[2].plot(epochs, history["val_dice"],   label="Val")
    axes[2].set_title("Dice Score (bg 제외)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curves.png"), dpi=150)
    plt.close()


# ────────────────────────────────────────────
# 6. Checkpoint 저장 helper
# ────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, epoch, metrics):
    torch.save({
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "optimizer"  : optimizer.state_dict(),
        **metrics,
    }, path)


# ────────────────────────────────────────────
# 7. Main
# ────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset & DataLoader ──────────────────
    train_dataset = SegDataset(os.path.join(args.data_root, "train"),
                               img_size=args.img_size, augment=True)
    val_dataset   = SegDataset(os.path.join(args.data_root, "tune"),
                               img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ── Model ─────────────────────────────────
    model = smp.Unet(
        encoder_name    = args.encoder,
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = args.num_classes,
    ).to(device)

    # ── Loss, Optimizer, Scheduler ───────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── History 초기화 ────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc" : [], "val_acc" : [],
        "train_dice": [], "val_dice": [],
    }

    os.makedirs(args.save_dir, exist_ok=True)

    best_dice     = 0.0
    best_acc      = 0.0
    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]  lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss, train_acc, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.num_classes)
        val_loss, val_acc, val_dice = validate(
            model, val_loader, criterion, device, args.num_classes)
        scheduler.step()

        # history 기록
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        print(f"  train → loss={train_loss:.4f}  acc={train_acc:.4f}  dice={train_dice:.4f}")
        print(f"  val   → loss={val_loss:.4f}  acc={val_acc:.4f}  dice={val_dice:.4f}")

        metrics = {"val_loss": val_loss, "val_acc": val_acc, "val_dice": val_dice}

        # ── Best Dice checkpoint ───────────────
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(os.path.join(args.save_dir, "best_dice.pth"),
                            model, optimizer, epoch, metrics)
            print(f"  ✓ best_dice.pth     saved  (dice={best_dice:.4f})")

        # ── Best Pixel Accuracy checkpoint ─────
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(os.path.join(args.save_dir, "best_acc.pth"),
                            model, optimizer, epoch, metrics)
            print(f"  ✓ best_acc.pth      saved  (acc={best_acc:.4f})")

        # ── Best Validation Loss checkpoint ────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(os.path.join(args.save_dir, "best_val_loss.pth"),
                            model, optimizer, epoch, metrics)
            print(f"  ✓ best_val_loss.pth saved  (loss={best_val_loss:.4f})")

        # ── 주기적 체크포인트 ──────────────────
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"epoch_{epoch:03d}.pth"))

        # ── curve PNG 매 epoch 갱신 ────────────
        save_curves(history, args.save_dir)

    print(f"\nDone!")
    print(f"  Best val Dice  : {best_dice:.4f}")
    print(f"  Best val Acc   : {best_acc:.4f}")
    print(f"  Best val Loss  : {best_val_loss:.4f}")
    print(f"  Curves saved → {os.path.join(args.save_dir, 'curves.png')}")


# ────────────────────────────────────────────
# 8. Argument parser
# ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMP UNet Training Script")

    # 경로
    parser.add_argument("--data_root", type=str, default="TEP_dataset/ro",
                        help="데이터 루트 (train/ val/ 하위 폴더)")
    parser.add_argument("--save_dir",  type=str, default="260401_ro")

    # 모델
    parser.add_argument("--encoder",     type=str, default="resnet50")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="배경 포함 클래스 수")
    parser.add_argument("--img_size",    type=int, default=512)

    # 학습
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--save_every",   type=int,   default=100)

    args = parser.parse_args()
    main(args)
