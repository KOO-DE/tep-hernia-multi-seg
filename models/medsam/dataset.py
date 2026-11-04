# -*- coding: utf-8 -*-
"""Dataset utilities for MedSAM multi-class TEP segmentation.

Design goals:
- images/masks layout by default; imgs/gts fallback
- full-image prompt by default for automatic segmentation
- optional foreground-focused crop augmentation for better recall
- moderate photometric augmentation for endoscopic domain shift
- cached dataset statistics for class-weighting / weighted sampling
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

join = os.path.join

IMAGE_DIR_CANDIDATES = ("images", "imgs")
MASK_DIR_CANDIDATES = ("masks", "gts")



def _resolve_existing_dir(root: str, candidates: Sequence[str]) -> str:
    for name in candidates:
        path = join(root, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Could not find any of {list(candidates)} under data root: {root}")



def _load_mask(mask_path: str) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path), dtype=np.int64)
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[..., 0]
        else:
            raise ValueError(f"Index mask should be single-channel, but got shape {mask.shape} for {mask_path}")
    return mask



def _safe_dtype_for_mask(max_label: int) -> np.dtype:
    return np.uint16 if max_label > 255 else np.uint8



def _compute_union_bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    fg = mask > 0
    if not fg.any():
        return None
    ys, xs = np.where(fg)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())



def _clamp_box(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return x0, y0, x1, y1



def _resize_image_and_mask(image: Image.Image, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    image_resized = image.resize((size, size), Image.BILINEAR)
    image_np = np.asarray(image_resized, dtype=np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))

    mask_dtype = _safe_dtype_for_mask(int(mask.max()) if mask.size else 0)
    mask_img = Image.fromarray(mask.astype(mask_dtype))
    mask_resized = np.asarray(mask_img.resize((size, size), Image.NEAREST), dtype=np.int64)
    return image_np, mask_resized



def _apply_gamma(image_np: np.ndarray, gamma: float) -> np.ndarray:
    image_np = np.clip(image_np, 0.0, 1.0)
    return np.power(image_np, gamma, dtype=np.float32)



@dataclass
class DatasetStats:
    num_images: int
    pixel_counts: List[int]
    image_counts: List[int]
    per_image_fg_classes: List[List[int]]


class SurgicalIndexDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        image_size: int = 1024,
        bbox_shift: int = 20,
        num_classes: int = 5,
        img_suffix: str = ".png",
        mask_suffix: str = ".png",
        prompt_mode: str = "full",
        full_box_margin: int = 0,
        train: bool = False,
        roi_crop_prob: float = 0.0,
        roi_context_min: float = 0.15,
        roi_context_max: float = 0.45,
        hflip_prob: float = 0.5,
        brightness: float = 0.15,
        contrast: float = 0.15,
        saturation: float = 0.10,
        gamma_min: float = 0.90,
        gamma_max: float = 1.10,
        blur_prob: float = 0.10,
        noise_prob: float = 0.10,
        cache_stats: bool = True,
    ):
        self.data_root = data_root
        self.img_path = _resolve_existing_dir(data_root, IMAGE_DIR_CANDIDATES)
        self.gt_path = _resolve_existing_dir(data_root, MASK_DIR_CANDIDATES)
        self.image_size = int(image_size)
        self.bbox_shift = int(bbox_shift)
        self.num_classes = int(num_classes)
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.prompt_mode = prompt_mode
        self.full_box_margin = int(full_box_margin)
        self.train = bool(train)

        self.roi_crop_prob = float(roi_crop_prob)
        self.roi_context_min = float(roi_context_min)
        self.roi_context_max = float(roi_context_max)
        self.hflip_prob = float(hflip_prob)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.blur_prob = float(blur_prob)
        self.noise_prob = float(noise_prob)
        self.cache_stats = bool(cache_stats)

        if self.prompt_mode not in {"full", "gt_bbox"}:
            raise ValueError(f"Unsupported prompt_mode={self.prompt_mode}. Use 'full' or 'gt_bbox'.")

        self.gt_files = sorted(glob.glob(join(self.gt_path, f"*{self.mask_suffix}")))
        self.gt_files = [
            f
            for f in self.gt_files
            if os.path.isfile(join(self.img_path, os.path.basename(f).replace(self.mask_suffix, self.img_suffix)))
        ]
        if len(self.gt_files) == 0:
            raise RuntimeError(f"No mask files found in {self.gt_path} with suffix {self.mask_suffix}")

        self.image_files = [join(self.img_path, os.path.basename(f).replace(self.mask_suffix, self.img_suffix)) for f in self.gt_files]

        print(
            f"[{os.path.basename(self.data_root)}] number of samples: {len(self.gt_files)} | "
            f"image_dir={self.img_path} | mask_dir={self.gt_path} | prompt_mode={self.prompt_mode} | train={self.train}"
        )

        self._stats: DatasetStats | None = None
        self._per_image_presence: List[List[int]] | None = None

    def __len__(self) -> int:
        return len(self.gt_files)

    @property
    def stats_cache_path(self) -> str:
        return join(self.data_root, f"dataset_stats_num_classes{self.num_classes}.json")

    def get_stats(self) -> DatasetStats:
        if self._stats is not None:
            return self._stats

        if self.cache_stats and os.path.isfile(self.stats_cache_path):
            try:
                with open(self.stats_cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._stats = DatasetStats(
                    num_images=int(data["num_images"]),
                    pixel_counts=[int(x) for x in data["pixel_counts"]],
                    image_counts=[int(x) for x in data["image_counts"]],
                    per_image_fg_classes=[[int(v) for v in row] for row in data["per_image_fg_classes"]],
                )
                self._per_image_presence = self._stats.per_image_fg_classes
                return self._stats
            except Exception:
                pass

        pixel_counts = np.zeros(self.num_classes, dtype=np.int64)
        image_counts = np.zeros(self.num_classes, dtype=np.int64)
        per_image_fg_classes: List[List[int]] = []

        for mask_path in self.gt_files:
            mask = _load_mask(mask_path)
            uniq, counts = np.unique(mask, return_counts=True)
            fg_classes = []
            for c, cnt in zip(uniq.tolist(), counts.tolist()):
                if c < 0 or c >= self.num_classes:
                    raise ValueError(
                        f"Mask values out of range in {mask_path}. Found class {c}, expected [0, {self.num_classes - 1}]"
                    )
                pixel_counts[c] += int(cnt)
                image_counts[c] += 1
                if c > 0:
                    fg_classes.append(int(c))
            per_image_fg_classes.append(sorted(fg_classes))

        self._stats = DatasetStats(
            num_images=len(self.gt_files),
            pixel_counts=pixel_counts.tolist(),
            image_counts=image_counts.tolist(),
            per_image_fg_classes=per_image_fg_classes,
        )
        self._per_image_presence = per_image_fg_classes

        if self.cache_stats:
            try:
                with open(self.stats_cache_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "num_images": self._stats.num_images,
                            "pixel_counts": self._stats.pixel_counts,
                            "image_counts": self._stats.image_counts,
                            "per_image_fg_classes": self._stats.per_image_fg_classes,
                        },
                        f,
                        indent=2,
                    )
            except Exception:
                pass

        return self._stats

    def get_class_pixel_counts(self) -> np.ndarray:
        return np.asarray(self.get_stats().pixel_counts, dtype=np.float64)

    def get_class_image_counts(self) -> np.ndarray:
        return np.asarray(self.get_stats().image_counts, dtype=np.float64)

    def get_image_sampling_weights(
        self,
        rare_power: float = 1.0,
        background_weight: float = 0.3,
    ) -> np.ndarray:
        stats = self.get_stats()
        image_counts = np.asarray(stats.image_counts, dtype=np.float64)
        image_counts = np.maximum(image_counts, 1.0)
        max_fg = image_counts[1:].max() if self.num_classes > 1 else image_counts.max()

        class_scores = np.ones(self.num_classes, dtype=np.float64)
        if self.num_classes > 1:
            class_scores[1:] = (max_fg / image_counts[1:]) ** float(rare_power)
            class_scores[0] = float(background_weight)

        weights = np.zeros(len(self.gt_files), dtype=np.float64)
        for i, fg_classes in enumerate(stats.per_image_fg_classes):
            if len(fg_classes) == 0:
                weights[i] = class_scores[0]
            else:
                weights[i] = float(np.mean([class_scores[c] for c in fg_classes]))

        weights = np.maximum(weights, 1e-8)
        weights = weights / weights.mean()
        return weights

    def __getitem__(self, index: int):
        mask_path = self.gt_files[index]
        image_path = self.image_files[index]
        base_name = os.path.basename(mask_path)

        image = Image.open(image_path).convert("RGB")
        mask = _load_mask(mask_path)

        uniq = np.unique(mask)
        if uniq.min() < 0 or uniq.max() >= self.num_classes:
            raise ValueError(
                f"Mask values out of range in {mask_path}. Found {uniq.tolist()}, expected [0, {self.num_classes - 1}]"
            )

        if self.train and self.roi_crop_prob > 0.0:
            image, mask = self._maybe_roi_crop(image, mask)

        if self.train:
            image, mask = self._apply_train_augmentations(image, mask)

        image_np, mask_resized = _resize_image_and_mask(image, mask, self.image_size)

        if self.prompt_mode == "full":
            bbox = self._full_image_box(self.image_size, self.image_size)
        else:
            bbox = self._bbox_from_foreground_union(mask_resized)

        return (
            torch.tensor(image_np, dtype=torch.float32),
            torch.tensor(mask_resized, dtype=torch.long),
            torch.tensor(bbox, dtype=torch.float32),
            base_name,
        )

    def _maybe_roi_crop(self, image: Image.Image, mask: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        if random.random() >= self.roi_crop_prob:
            return image, mask

        bbox = _compute_union_bbox(mask)
        if bbox is None:
            return image, mask

        x0, y0, x1, y1 = bbox
        h, w = mask.shape
        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)

        ctx_ratio = random.uniform(self.roi_context_min, self.roi_context_max)
        ctx_x = int(round(bw * ctx_ratio))
        ctx_y = int(round(bh * ctx_ratio))

        jitter_x = int(round(bw * random.uniform(-0.10, 0.10)))
        jitter_y = int(round(bh * random.uniform(-0.10, 0.10)))

        crop_x0 = x0 - ctx_x + jitter_x
        crop_y0 = y0 - ctx_y + jitter_y
        crop_x1 = x1 + 1 + ctx_x + jitter_x
        crop_y1 = y1 + 1 + ctx_y + jitter_y

        min_side = int(max(bw, bh) * random.uniform(1.15, 1.80))
        cx = 0.5 * (crop_x0 + crop_x1)
        cy = 0.5 * (crop_y0 + crop_y1)
        crop_x0 = int(round(cx - min_side / 2))
        crop_x1 = int(round(cx + min_side / 2))
        crop_y0 = int(round(cy - min_side / 2))
        crop_y1 = int(round(cy + min_side / 2))

        crop_x0, crop_y0, crop_x1, crop_y1 = _clamp_box(crop_x0, crop_y0, crop_x1, crop_y1, w, h)

        image = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        mask = mask[crop_y0:crop_y1, crop_x0:crop_x1]
        return image, mask

    def _apply_train_augmentations(self, image: Image.Image, mask: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        if random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.ascontiguousarray(np.fliplr(mask))

        if self.brightness > 0:
            factor = random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)

        if self.contrast > 0:
            factor = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)

        if self.saturation > 0:
            factor = random.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation)
            image = ImageEnhance.Color(image).enhance(factor)

        if self.blur_prob > 0 and random.random() < self.blur_prob:
            radius = random.uniform(0.3, 1.2)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        image_np = np.asarray(image, dtype=np.float32) / 255.0

        if self.gamma_min != 1.0 or self.gamma_max != 1.0:
            gamma = random.uniform(self.gamma_min, self.gamma_max)
            image_np = _apply_gamma(image_np, gamma)

        if self.noise_prob > 0 and random.random() < self.noise_prob:
            sigma = random.uniform(0.005, 0.02)
            noise = np.random.normal(loc=0.0, scale=sigma, size=image_np.shape).astype(np.float32)
            image_np = np.clip(image_np + noise, 0.0, 1.0)

        image = Image.fromarray((np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8))
        return image, mask

    def _full_image_box(self, height: int, width: int) -> np.ndarray:
        margin = max(0, int(self.full_box_margin))
        x_min = margin
        y_min = margin
        x_max = width - 1 - margin
        y_max = height - 1 - margin
        if x_max <= x_min or y_max <= y_min:
            x_min, y_min, x_max, y_max = 0, 0, width - 1, height - 1
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    def _bbox_from_foreground_union(self, mask: np.ndarray) -> np.ndarray:
        bbox = _compute_union_bbox(mask)
        h, w = mask.shape
        if bbox is None:
            return self._full_image_box(h, w)
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(w - 1, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(h - 1, y_max + random.randint(0, self.bbox_shift))
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
