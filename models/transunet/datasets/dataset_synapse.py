import os
import random
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms

# ── Color Map ─────────────────────────────────────────────
COLOR_MAP = {
    (0, 0, 0):       0,  # Background
    (231, 226, 203): 1,  # Inf. epigastric vv.
    (252, 155, 192): 2,  # Pubic arch
    (176, 145, 131): 3,  # Spermatic cord
    (169, 117, 87):  4,  # Vas deferens
}


# ── Augmentation ───────────────────────────────────────────
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y, c = image.shape  # RGB: (H, W, 3)
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # (H,W,3) → (3,H,W)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

# ── Dataset ────────────────────────────────────────────────
class TEP_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.img_dir = os.path.join(base_dir, split, 'images')
        self.mask_dir = os.path.join(base_dir, split, 'masks')

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx].strip('\n')
        img = np.array(Image.open(os.path.join(self.img_dir, name + '.png')).convert('RGB')) / 255.0
        label = np.array(Image.open(os.path.join(self.mask_dir, name + '.png')).convert('L'))

        sample = {'image': img, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        else:
            # transform 없을 때도 (H,W,3) → (3,H,W) 변환 필요
            img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': img, 'label': label}

        sample['case_name'] = name
        return sample
