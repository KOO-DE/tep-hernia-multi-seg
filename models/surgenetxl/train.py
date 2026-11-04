import os
import cv2
import math
import glob
import time
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from metaformer import MetaFormerFPN


# ----------------------------
# Utils
# ----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        distributed = True
    else:
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        distributed = False
    return local_rank, distributed


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def reduce_tensor(tensor: torch.Tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    rt = tensor.detach().clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_curve(train_values, val_values, ylabel, save_path):
    plt.figure(figsize=(8, 6))
    epochs = list(range(1, len(train_values) + 1))
    plt.plot(epochs, train_values, label=f"Train {ylabel}")
    plt.plot(epochs, val_values, label=f"Val {ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_all_curves(history, save_dir):
    plot_curve(history["train_loss"], history["val_loss"], "Loss",
               os.path.join(save_dir, "loss_curve.png"))
    plot_curve(history["train_dice"], history["val_dice"], "Dice",
               os.path.join(save_dir, "dice_curve.png"))
    plot_curve(history["train_iou"], history["val_iou"], "IoU",
               os.path.join(save_dir, "iou_curve.png"))
    plot_curve(history["train_pixel_acc"], history["val_pixel_acc"], "Pixel Accuracy",
               os.path.join(save_dir, "pixel_acc_curve.png"))


# ----------------------------
# Dataset
# ----------------------------
class SegDataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        image_size=512,
        num_classes=5,
        is_train=True,
        check_mask_values=True,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.num_classes = num_classes
        self.is_train = is_train
        self.check_mask_values = check_mask_values

        exts = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
        self.img_paths = []
        for ext in exts:
            self.img_paths.extend(glob.glob(os.path.join(img_dir, f"*.{ext}")))
        self.img_paths = sorted(self.img_paths)

        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {img_dir}")

        self.pairs = []
        for img_path in self.img_paths:
            name = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {img_path}: {mask_path}")
            self.pairs.append((img_path, mask_path))

    def __len__(self):
        return len(self.pairs)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        if random.random() < 0.3:
            angle = random.choice([90, 180, 270])
            if angle == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
                mask = cv2.rotate(mask, cv2.ROTATE_180)
            elif angle == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if random.random() < 0.3:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-10, 10)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return image, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_train:
            image, mask = self._augment(image, mask)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.check_mask_values:
            unique_vals = np.unique(mask)
            if unique_vals.min() < 0 or unique_vals.max() >= self.num_classes:
                raise ValueError(
                    f"Invalid mask values in {mask_path}. "
                    f"Expected values in [0, {self.num_classes - 1}], but got {unique_vals.tolist()}"
                )

        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        image = np.transpose(image, (2, 0, 1))

        mask = mask.astype(np.int64)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return image, mask


# ----------------------------
# Loss / Metrics
# ----------------------------
class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(
            torch.clamp(targets, min=0),
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        dice_losses = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue

            prob_cls = probs[:, cls]
            target_cls = targets_onehot[:, cls]

            intersection = (prob_cls * target_cls).sum(dim=(1, 2))
            denom = prob_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
            dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
            dice_losses.append(1.0 - dice.mean())

        if len(dice_losses) == 0:
            return logits.new_tensor(0.0)

        return torch.stack(dice_losses).mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        ce_weight=0.5,
        dice_weight=0.5,
        use_dice=True,
        ignore_index=None,
    ):
        super().__init__()
        if ignore_index is None:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.use_dice = use_dice
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        if use_dice:
            dice_ignore_index = 0 if ignore_index is None else ignore_index
            self.dice = MultiClassDiceLoss(
                num_classes=num_classes,
                ignore_index=dice_ignore_index,
            )

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        if not self.use_dice:
            return ce_loss
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


@torch.no_grad()
def dice_score_multiclass(logits, targets, num_classes, ignore_index=0, eps=1e-7):
    preds = torch.argmax(logits, dim=1)

    dice_per_class = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
        dice = (2 * intersection + eps) / (union + eps)
        dice_per_class.append(dice.mean())

    if len(dice_per_class) == 0:
        return torch.tensor(0.0, device=logits.device)

    return torch.stack(dice_per_class).mean()


@torch.no_grad()
def iou_score_multiclass(logits, targets, num_classes, ignore_index=0, eps=1e-7):
    preds = torch.argmax(logits, dim=1)

    iou_per_class = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2)) - intersection
        iou = (intersection + eps) / (union + eps)
        iou_per_class.append(iou.mean())

    if len(iou_per_class) == 0:
        return torch.tensor(0.0, device=logits.device)

    return torch.stack(iou_per_class).mean()


@torch.no_grad()
def pixel_accuracy_multiclass(logits, targets, ignore_index=None):
    preds = torch.argmax(logits, dim=1)

    if ignore_index is None:
        correct = (preds == targets).float().mean()
    else:
        valid = (targets != ignore_index)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        correct = ((preds == targets) & valid).float().sum() / valid.float().sum()

    return correct


# ----------------------------
# Tuning helpers
# ----------------------------
def set_requires_grad(module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def freeze_head_only(model):
    if hasattr(model, "metaformer") and hasattr(model.metaformer, "head"):
        set_requires_grad(model.metaformer.head, False)


def apply_tuning_mode(model, tuning_mode: str):
    """
    tuning_mode:
      - full
      - decoder_only
      - last2_and_decoder
    """
    if not hasattr(model, "metaformer"):
        raise ValueError("Model does not have metaformer backbone.")

    metaformer = model.metaformer

    # default: freeze everything in backbone first
    set_requires_grad(metaformer, False)

    # classification head is never used for segmentation
    if hasattr(metaformer, "head"):
        set_requires_grad(metaformer.head, False)

    # decoder is always trainable
    if hasattr(model, "FPN"):
        set_requires_grad(model.FPN, True)

    if tuning_mode == "full":
        set_requires_grad(metaformer, True)
        if hasattr(metaformer, "head"):
            set_requires_grad(metaformer.head, False)

    elif tuning_mode == "decoder_only":
        # keep backbone frozen, decoder trainable
        pass

    elif tuning_mode == "last2_and_decoder":
        # train last 2 encoder stages + corresponding downsample layers + final norm
        # MetaFormer structure in metaformer.py:
        # downsample_layers: 4
        # stages: 4
        if not hasattr(metaformer, "stages") or not hasattr(metaformer, "downsample_layers"):
            raise ValueError("metaformer does not expose stages/downsample_layers as expected.")

        num_stages = len(metaformer.stages)
        if num_stages < 4:
            raise ValueError(f"Expected at least 4 stages, got {num_stages}")

        # open last two stages
        set_requires_grad(metaformer.stages[2], True)
        set_requires_grad(metaformer.stages[3], True)

        # open the downsample layers feeding those stages
        # downsample_layers[2] feeds stage 2, downsample_layers[3] feeds stage 3
        set_requires_grad(metaformer.downsample_layers[2], True)
        set_requires_grad(metaformer.downsample_layers[3], True)

        # final norm before head/features
        if hasattr(metaformer, "norm"):
            set_requires_grad(metaformer.norm, True)

    else:
        raise ValueError(
            f"Unsupported tuning_mode: {tuning_mode}. "
            f"Choose from ['full', 'decoder_only', 'last2_and_decoder']"
        )


def summarize_trainable_params(model, is_main=True):
    if not is_main:
        return
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / max(total,1):.2f}%)")


# ----------------------------
# Train / Validate
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, num_classes, metric_ignore_index=0):
    model.train()

    running_loss = torch.tensor(0.0, device=device)
    running_dice = torch.tensor(0.0, device=device)
    running_iou = torch.tensor(0.0, device=device)
    running_pixel_acc = torch.tensor(0.0, device=device)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_dice = dice_score_multiclass(
            logits, masks, num_classes=num_classes, ignore_index=metric_ignore_index
        )
        batch_iou = iou_score_multiclass(
            logits, masks, num_classes=num_classes, ignore_index=metric_ignore_index
        )
        batch_pixel_acc = pixel_accuracy_multiclass(logits, masks, ignore_index=None)

        running_loss += loss.detach()
        running_dice += batch_dice.detach()
        running_iou += batch_iou.detach()
        running_pixel_acc += batch_pixel_acc.detach()

    running_loss = reduce_tensor(running_loss / len(loader))
    running_dice = reduce_tensor(running_dice / len(loader))
    running_iou = reduce_tensor(running_iou / len(loader))
    running_pixel_acc = reduce_tensor(running_pixel_acc / len(loader))

    return (
        running_loss.item(),
        running_dice.item(),
        running_iou.item(),
        running_pixel_acc.item(),
    )


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, metric_ignore_index=0):
    model.eval()

    running_loss = torch.tensor(0.0, device=device)
    running_dice = torch.tensor(0.0, device=device)
    running_iou = torch.tensor(0.0, device=device)
    running_pixel_acc = torch.tensor(0.0, device=device)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

        loss = criterion(logits, masks)

        batch_dice = dice_score_multiclass(
            logits, masks, num_classes=num_classes, ignore_index=metric_ignore_index
        )
        batch_iou = iou_score_multiclass(
            logits, masks, num_classes=num_classes, ignore_index=metric_ignore_index
        )
        batch_pixel_acc = pixel_accuracy_multiclass(logits, masks, ignore_index=None)

        running_loss += loss.detach()
        running_dice += batch_dice.detach()
        running_iou += batch_iou.detach()
        running_pixel_acc += batch_pixel_acc.detach()

    running_loss = reduce_tensor(running_loss / len(loader))
    running_dice = reduce_tensor(running_dice / len(loader))
    running_iou = reduce_tensor(running_iou / len(loader))
    running_pixel_acc = reduce_tensor(running_pixel_acc / len(loader))

    return (
        running_loss.item(),
        running_dice.item(),
        running_iou.item(),
        running_pixel_acc.item(),
    )


# ----------------------------
# Main
# ----------------------------
def main(args):
    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script expects at least one GPU.")

    local_rank, distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)

    train_dataset = SegDataset(
        img_dir=args.train_img_dir,
        mask_dir=args.train_mask_dir,
        image_size=args.image_size,
        num_classes=args.num_classes,
        is_train=True,
        check_mask_values=not args.disable_mask_check,
    )
    val_dataset = SegDataset(
        img_dir=args.val_img_dir,
        mask_dir=args.val_mask_dir,
        image_size=args.image_size,
        num_classes=args.num_classes,
        is_train=False,
        check_mask_values=not args.disable_mask_check,
    )

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    ckpt_urls = {
        "SurgeNetXL": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetXL_checkpoint_epoch0050_teacher.pth?download=true",
        "SurgeNet": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNet_checkpoint_epoch0050_teacher.pth?download=true",
        "SurgeNet-Small": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/SurgeNetSmall_checkpoint_epoch0050_teacher.pth?download=true",
        "SurgeNet-Public": "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/Public_checkpoint0050.pth?download=true",
    }

    pretrained_source = ckpt_urls[args.pretrained_ckpt] if args.pretrained_ckpt in ckpt_urls else args.pretrained_ckpt

    model = MetaFormerFPN(
        num_classes=args.num_classes,
        pretrained="SurgeNet",
        pretrained_weights=pretrained_source,
        verbose=is_main_process(),
    )

    apply_tuning_mode(model, args.tuning_mode)
    summarize_trainable_params(model, is_main=is_main_process())

    if is_main_process():
        print(f"Tuning mode: {args.tuning_mode}")

    model = model.to(device)

    start_epoch = 0
    best_dice = -1.0
    best_val_loss = float("inf")
    best_pixel_acc = -1.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_iou": [],
        "val_iou": [],
        "train_pixel_acc": [],
        "val_pixel_acc": [],
    }

    checkpoint = None
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_dice = checkpoint.get("best_dice", -1.0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_pixel_acc = checkpoint.get("best_pixel_acc", -1.0)
        history = checkpoint.get("history", history)

        if is_main_process():
            print(f"Loaded checkpoint: {args.resume}")

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.find_unused_parameters,
        )

    model_for_optim = model.module if distributed else model

    loss_ignore_index = None if args.ignore_index < 0 else args.ignore_index
    metric_ignore_index = args.metric_ignore_index

    criterion = CombinedLoss(
        num_classes=args.num_classes,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        use_dice=args.use_dice,
        ignore_index=loss_ignore_index,
    )

    encoder_params = [p for p in model_for_optim.metaformer.parameters() if p.requires_grad]
    decoder_params = [p for p in model_for_optim.FPN.parameters() if p.requires_grad]

    param_groups = []
    if len(encoder_params) > 0:
        param_groups.append({"params": encoder_params, "lr": args.encoder_lr})
    if len(decoder_params) > 0:
        param_groups.append({"params": decoder_params, "lr": args.decoder_lr})

    if len(param_groups) == 0:
        raise ValueError("No trainable parameters found. Check tuning_mode settings.")

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = args.epochs
    warmup_steps = args.warmup_epochs

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if checkpoint is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        t0 = time.time()

        train_loss, train_dice, train_iou, train_pixel_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_classes=args.num_classes,
            metric_ignore_index=metric_ignore_index,
        )

        val_loss, val_dice, val_iou, val_pixel_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=args.num_classes,
            metric_ignore_index=metric_ignore_index,
        )

        elapsed = time.time() - t0

        improved_dice = val_dice > best_dice
        improved_loss = val_loss < best_val_loss
        improved_pixel_acc = val_pixel_acc > best_pixel_acc

        if improved_dice:
            best_dice = val_dice
        if improved_loss:
            best_val_loss = val_loss
        if improved_pixel_acc:
            best_pixel_acc = val_pixel_acc

        if is_main_process():
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_dice"].append(train_dice)
            history["val_dice"].append(val_dice)
            history["train_iou"].append(train_iou)
            history["val_iou"].append(val_iou)
            history["train_pixel_acc"].append(train_pixel_acc)
            history["val_pixel_acc"].append(val_pixel_acc)

            current_lrs = [pg["lr"] for pg in optimizer.param_groups]
            if len(current_lrs) == 1:
                lr_msg = f"lr={current_lrs[0]:.2e}"
            else:
                lr_msg = f"enc_lr={current_lrs[0]:.2e} dec_lr={current_lrs[1]:.2e}"

            print(
                f"[Epoch {epoch+1:03d}/{args.epochs:03d}] "
                f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} train_iou={train_iou:.4f} train_pixel_acc={train_pixel_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f} val_pixel_acc={val_pixel_acc:.4f} | "
                f"{lr_msg} | "
                f"time={elapsed:.1f}s"
            )

            model_state = model.module.state_dict() if distributed else model.state_dict()

            ckpt = {
                "epoch": epoch,
                "model": model_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_dice": best_dice,
                "best_val_loss": best_val_loss,
                "best_pixel_acc": best_pixel_acc,
                "history": history,
                "args": vars(args),
            }

            torch.save(ckpt, os.path.join(args.save_dir, "latest.pth"))
            save_json(history, os.path.join(args.save_dir, "history.json"))
            save_all_curves(history, args.save_dir)

            if improved_dice:
                torch.save(ckpt, os.path.join(args.save_dir, "best_dice.pth"))
                print(f"Saved best_dice.pth (val_dice={best_dice:.4f})")

            if improved_loss:
                torch.save(ckpt, os.path.join(args.save_dir, "best_val_loss.pth"))
                print(f"Saved best_val_loss.pth (val_loss={best_val_loss:.4f})")

            if improved_pixel_acc:
                torch.save(ckpt, os.path.join(args.save_dir, "best_pixel_acc.pth"))
                print(f"Saved best_pixel_acc.pth (val_pixel_acc={best_pixel_acc:.4f})")

        scheduler.step()

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SurgeNet + FPN (1GPU or DDP)")

    parser.add_argument("--train_img_dir", type=str, default="TEP_dataset/ro/train/images")
    parser.add_argument("--train_mask_dir", type=str, default="TEP_dataset/ro/train/masks")
    parser.add_argument("--val_img_dir", type=str, default="TEP_dataset/ro/tune/images")
    parser.add_argument("--val_mask_dir", type=str, default="TEP_dataset/ro/tune/masks")

    parser.add_argument("--save_dir", type=str, default="checkpoint_tep/260401_ro_xl_512")
    parser.add_argument("--resume", type=str, default="")

    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default="SurgeNetXL",
        help="Pretrained checkpoint key or a local/remote checkpoint path"
    )

    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--decoder_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--tuning_mode",
        type=str,
        default="full",
        choices=["full", "decoder_only", "last2_and_decoder"],
        help="Fine-tuning strategy"
    )
    parser.add_argument("--find_unused_parameters", action="store_true")
    parser.add_argument("--disable_mask_check", action="store_true")

    parser.add_argument("--use_dice", action="store_true")
    parser.add_argument("--ce_weight", type=float, default=0.5)
    parser.add_argument("--dice_weight", type=float, default=0.5)

    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--metric_ignore_index", type=int, default=0)

    args = parser.parse_args()
    main(args)
