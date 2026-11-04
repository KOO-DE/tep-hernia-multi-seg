import numpy as np
import os
import torch
import torch.nn as nn
from medpy import metric
from scipy.ndimage import zoom
from PIL import Image


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), \
            'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def colorize_mask(mask):
    color_map = {
        0: (0, 0, 0),        # Background
        1: (231, 226, 203),  # Inf. epigastric vv.
        2: (252, 155, 192),  # Pubic arch
        3: (176, 145, 131),  # Spermatic cord
        4: (169, 117, 87),   # Vas deferens
    }

    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_idx, color in color_map.items():
        colored[mask == cls_idx] = color
    return colored


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    # RGB image: (3, H, W)
    if len(image.shape) == 3:
        c, x, y = image.shape

        if x != patch_size[0] or y != patch_size[1]:
            image_resized = np.stack([
                zoom(image[i], (patch_size[0] / x, patch_size[1] / y), order=3)
                for i in range(c)
            ], axis=0)
        else:
            image_resized = image

        input_tensor = torch.from_numpy(image_resized).unsqueeze(0).float().cuda()

        net.eval()
        with torch.no_grad():
            outputs = net(input_tensor)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                prediction = out

    # grayscale image: (H, W)
    else:
        x, y = image.shape

        if x != patch_size[0] or y != patch_size[1]:
            image_resized = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        else:
            image_resized = image

        input_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float().cuda()

        net.eval()
        with torch.no_grad():
            outputs = net(input_tensor)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                prediction = out

    prediction = prediction.astype(np.uint8)
    label = label.astype(np.uint8)

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None and case is not None:
        os.makedirs(test_save_path, exist_ok=True)

        pred_dir = os.path.join(test_save_path, "pred")
        pred_overlay_dir = os.path.join(test_save_path, "pred_overlay")
        gt_overlay_dir = os.path.join(test_save_path, "gt_overlay")

        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(pred_overlay_dir, exist_ok=True)
        os.makedirs(gt_overlay_dir, exist_ok=True)

        pred_colored = colorize_mask(prediction)
        gt_colored = colorize_mask(label)

        if len(image.shape) == 3:
            image_vis = np.transpose(image, (1, 2, 0))
            image_vis = (image_vis * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image_vis = (image * 255.0).clip(0, 255).astype(np.uint8)
            image_vis = np.stack([image_vis] * 3, axis=-1)

        alpha = 0.5
        pred_overlay = (alpha * image_vis + (1 - alpha) * pred_colored).clip(0, 255).astype(np.uint8)
        gt_overlay = (alpha * image_vis + (1 - alpha) * gt_colored).clip(0, 255).astype(np.uint8)

        Image.fromarray(pred_colored).save(os.path.join(pred_dir, case + ".png"))
        Image.fromarray(pred_overlay).save(os.path.join(pred_overlay_dir, case + ".png"))
        Image.fromarray(gt_overlay).save(os.path.join(gt_overlay_dir, case + ".png"))

    return metric_list
