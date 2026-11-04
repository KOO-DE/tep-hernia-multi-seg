import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from medpy.metric.binary import dc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================================================
# Loss helpers
# =========================================================
def get_one_hot(label, num_classes):
    # label: [B, H, W]
    return F.one_hot(label.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


def edge_map_from_mask(mask):
    """
    mask: [B, C, H, W] in {0,1} or soft prob
    returns: edge-like map [B, C, H, W]
    """
    # morphological gradient approximation
    maxpool = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    minpool = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    edge = (maxpool - minpool).clamp(min=0.0, max=1.0)
    return edge


class BoundaryLoss(nn.Module):
    """
    Boundary loss via BCE between predicted edge map and GT edge map.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, logits, target, num_classes):
        # logits: [B, C, H, W]
        # target: [B, H, W]
        probs = torch.softmax(logits, dim=1)
        target_oh = get_one_hot(target, num_classes)

        pred_edge = edge_map_from_mask(probs)
        gt_edge = edge_map_from_mask(target_oh)

        return self.bce(pred_edge, gt_edge)


# ---------- soft clDice helpers ----------
def soft_erode(img):
    if img.dim() == 4:
        p1 = -F.max_pool2d(-img, (3, 1), stride=(1, 1), padding=(1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), stride=(1, 1), padding=(0, 1))
        return torch.min(p1, p2)
    raise ValueError("soft_erode expects 4D tensor [B,C,H,W]")


def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iters=10):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftClDiceLoss(nn.Module):
    """
    Soft clDice for selected classes only.
    """
    def __init__(self, iter_num=10, smooth=1e-5):
        super().__init__()
        self.iter_num = iter_num
        self.smooth = smooth

    def forward(self, logits, target, class_ids, num_classes):
        # logits: [B, C, H, W]
        # target: [B, H, W]
        probs = torch.softmax(logits, dim=1)
        target_oh = get_one_hot(target, num_classes)

        probs = probs[:, class_ids, ...]
        target_oh = target_oh[:, class_ids, ...]

        pred_skel = soft_skel(probs, self.iter_num)
        gt_skel = soft_skel(target_oh, self.iter_num)

        tprec = (torch.sum(pred_skel * target_oh) + self.smooth) / (torch.sum(pred_skel) + self.smooth)
        tsens = (torch.sum(gt_skel * probs) + self.smooth) / (torch.sum(gt_skel) + self.smooth)

        cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + self.smooth)
        return 1.0 - cl_dice


def build_ce_loss(args):
    if args.loss_mode == "weighted_ce_dice":
        if args.class_weights is None or len(args.class_weights) != args.num_classes:
            raise ValueError(
                f"For weighted_ce_dice, args.class_weights must have length == num_classes ({args.num_classes})."
            )
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32).cuda()
        return CrossEntropyLoss(weight=class_weights)
    else:
        return CrossEntropyLoss()


def compute_total_loss(args, outputs, label_batch, ce_loss, dice_loss, boundary_loss_fn, cldice_loss_fn):
    """
    Returns:
        total_loss, loss_dict
    """
    loss_ce = ce_loss(outputs, label_batch.long())
    loss_dice = dice_loss(outputs, label_batch, softmax=True)

    total_loss = args.ce_weight * loss_ce + args.dice_weight * loss_dice
    loss_dict = {
        "ce": loss_ce,
        "dice": loss_dice,
        "boundary": torch.tensor(0.0, device=outputs.device),
        "cldice": torch.tensor(0.0, device=outputs.device),
    }

    if args.loss_mode in ["ce_dice_boundary", "ce_dice_boundary_cldice"]:
        loss_boundary = boundary_loss_fn(outputs, label_batch, args.num_classes)
        total_loss = total_loss + args.boundary_weight * loss_boundary
        loss_dict["boundary"] = loss_boundary

    if args.loss_mode == "ce_dice_boundary_cldice":
        if args.cldice_classes is None or len(args.cldice_classes) == 0:
            raise ValueError("For ce_dice_boundary_cldice, args.cldice_classes must be provided.")
        loss_cldice = cldice_loss_fn(
            outputs, label_batch, class_ids=args.cldice_classes, num_classes=args.num_classes
        )
        total_loss = total_loss + args.cldice_weight * loss_cldice
        loss_dict["cldice"] = loss_cldice

    return total_loss, loss_dict


# =========================================================
# Trainer
# =========================================================
def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import TEP_dataset, RandomGenerator

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = TEP_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size])]
        )
    )
    print("The length of train set is: {}".format(len(db_train)))

    db_val = TEP_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="tune",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size])]
        )
    )
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train, batch_size=batch_size, shuffle=True, num_workers=8,
        pin_memory=True, worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    ce_loss = build_ce_loss(args)
    dice_loss = DiceLoss(num_classes)
    boundary_loss_fn = BoundaryLoss()
    cldice_loss_fn = SoftClDiceLoss(iter_num=args.cldice_iter)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    best_val_loss = float('inf')
    best_dice = 0.0
    best_accuracy = 0.0

    train_loss_history = []
    train_dice_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_dice_history = []
    val_accuracy_history = []

    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        # ---------------- Train ----------------
        model.train()
        train_loss_epoch = 0.0
        train_dice_epoch = 0.0
        train_accuracy_epoch = 0.0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)

            loss, loss_dict = compute_total_loss(
                args=args,
                outputs=outputs,
                label_batch=label_batch,
                ce_loss=ce_loss,
                dice_loss=dice_loss,
                boundary_loss_fn=boundary_loss_fn,
                cldice_loss_fn=cldice_loss_fn
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            train_loss_epoch += loss.item()

            with torch.no_grad():
                pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                pred_np = pred.cpu().numpy()
                label_np = label_batch.cpu().numpy()

                dice_per_class = []
                for cls in range(1, num_classes):
                    pred_cls = (pred_np == cls)
                    label_cls = (label_np == cls)
                    if label_cls.sum() > 0:
                        dice_per_class.append(dc(pred_cls, label_cls))
                if dice_per_class:
                    train_dice_epoch += np.mean(dice_per_class)

                correct = (pred_np == label_np).sum()
                total = label_np.size
                train_accuracy_epoch += correct / total

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_ce', loss_dict["ce"].item(), iter_num)
            writer.add_scalar('info/loss_dice', loss_dict["dice"].item(), iter_num)
            writer.add_scalar('info/loss_boundary', loss_dict["boundary"].item(), iter_num)
            writer.add_scalar('info/loss_cldice', loss_dict["cldice"].item(), iter_num)

            logging.info(
                'iteration %d : total_loss: %f, ce: %f, dice: %f, boundary: %f, cldice: %f' % (
                    iter_num,
                    loss.item(),
                    loss_dict["ce"].item(),
                    loss_dict["dice"].item(),
                    loss_dict["boundary"].item(),
                    loss_dict["cldice"].item()
                )
            )

            if iter_num % 20 == 0:
                image = image_batch[0, 0:3, :, :]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                writer.add_image('train/Image', image, iter_num)
                outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        train_loss_history.append(train_loss_epoch / len(trainloader))
        train_dice_history.append(train_dice_epoch / len(trainloader))
        train_accuracy_history.append(train_accuracy_epoch / len(trainloader))

        # ---------------- Validation ----------------
        model.eval()
        val_loss_total = 0.0
        val_loss_ce_total = 0.0
        val_loss_dice_total = 0.0
        val_loss_boundary_total = 0.0
        val_loss_cldice_total = 0.0
        val_dice_total = 0.0
        val_accuracy_total = 0.0

        with torch.no_grad():
            for val_batch in valloader:
                val_image, val_label = val_batch['image'].cuda(), val_batch['label'].cuda()
                val_output = model(val_image)

                v_loss, v_loss_dict = compute_total_loss(
                    args=args,
                    outputs=val_output,
                    label_batch=val_label,
                    ce_loss=ce_loss,
                    dice_loss=dice_loss,
                    boundary_loss_fn=boundary_loss_fn,
                    cldice_loss_fn=cldice_loss_fn
                )

                val_loss_total += v_loss.item()
                val_loss_ce_total += v_loss_dict["ce"].item()
                val_loss_dice_total += v_loss_dict["dice"].item()
                val_loss_boundary_total += v_loss_dict["boundary"].item()
                val_loss_cldice_total += v_loss_dict["cldice"].item()

                pred = torch.argmax(torch.softmax(val_output, dim=1), dim=1)
                pred_np = pred.cpu().numpy()
                label_np = val_label.cpu().numpy()

                dice_per_class = []
                for cls in range(1, num_classes):
                    pred_cls = (pred_np == cls)
                    label_cls = (label_np == cls)
                    if label_cls.sum() > 0:
                        dice_per_class.append(dc(pred_cls, label_cls))
                if dice_per_class:
                    val_dice_total += np.mean(dice_per_class)

                correct = (pred_np == label_np).sum()
                total = label_np.size
                val_accuracy_total += correct / total

        val_loss_avg = val_loss_total / len(valloader)
        val_loss_ce_avg = val_loss_ce_total / len(valloader)
        val_loss_dice_avg = val_loss_dice_total / len(valloader)
        val_loss_boundary_avg = val_loss_boundary_total / len(valloader)
        val_loss_cldice_avg = val_loss_cldice_total / len(valloader)
        val_dice_avg = val_dice_total / len(valloader)
        val_accuracy_avg = val_accuracy_total / len(valloader)

        val_loss_history.append(val_loss_avg)
        val_dice_history.append(val_dice_avg)
        val_accuracy_history.append(val_accuracy_avg)

        writer.add_scalar('info/val_loss', val_loss_avg, epoch_num)
        writer.add_scalar('info/val_loss_ce', val_loss_ce_avg, epoch_num)
        writer.add_scalar('info/val_loss_dice', val_loss_dice_avg, epoch_num)
        writer.add_scalar('info/val_loss_boundary', val_loss_boundary_avg, epoch_num)
        writer.add_scalar('info/val_loss_cldice', val_loss_cldice_avg, epoch_num)
        writer.add_scalar('info/val_dice', val_dice_avg, epoch_num)
        writer.add_scalar('info/val_accuracy', val_accuracy_avg, epoch_num)

        logging.info(
            'epoch %d : val_loss: %f, val_loss_ce: %f, val_loss_dice: %f, val_loss_boundary: %f, val_loss_cldice: %f, val_dice: %f, val_accuracy: %f' % (
                epoch_num, val_loss_avg, val_loss_ce_avg, val_loss_dice_avg,
                val_loss_boundary_avg, val_loss_cldice_avg, val_dice_avg, val_accuracy_avg
            )
        )

        # ---------------- PNG save ----------------
        epochs_so_far = list(range(1, epoch_num + 2))
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        axes[0].plot(epochs_so_far, train_loss_history, label='Train Loss', color='blue')
        axes[0].plot(epochs_so_far, val_loss_history, label='Val Loss', color='orange')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(epochs_so_far, train_dice_history, label='Train Dice', color='blue')
        axes[1].plot(epochs_so_far, val_dice_history, label='Val Dice', color='orange')
        axes[1].set_title('Dice')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(epochs_so_far, train_accuracy_history, label='Train Accuracy', color='blue')
        axes[2].plot(epochs_so_far, val_accuracy_history, label='Val Accuracy', color='orange')
        axes[2].set_title('Accuracy')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(snapshot_path, 'training_curve.png'))
        plt.close()

        # ---------------- Best models ----------------
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            save_mode_path = os.path.join(snapshot_path, 'best_val_loss_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Best val_loss model saved at epoch {} | val_loss: {:.4f}".format(epoch_num, best_val_loss))

        if val_dice_avg > best_dice:
            best_dice = val_dice_avg
            save_mode_path = os.path.join(snapshot_path, 'best_dice_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Best dice model saved at epoch {} | val_dice: {:.4f}".format(epoch_num, best_dice))

        if val_accuracy_avg > best_accuracy:
            best_accuracy = val_accuracy_avg
            save_mode_path = os.path.join(snapshot_path, 'best_accuracy_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Best accuracy model saved at epoch {} | val_accuracy: {:.4f}".format(epoch_num, best_accuracy))

        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
