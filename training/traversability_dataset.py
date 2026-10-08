import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import cv2

IMG_SIZE = 224
DEPTH_MAX_M = 3.0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TraversabilityDataset(Dataset):
    def __init__(self, nyu_root, masks_root, split='train', file_list=None, augment=False):
        self.rgb_dir = Path(nyu_root).expanduser() / 'image' / split
        self.depth_dir = Path(nyu_root).expanduser() / 'depth' / split
        self.mask_dir = Path(masks_root).expanduser() / 'masks' / split
        self.augment = augment

        self.files = list(file_list) if file_list is not None else \
            sorted(p.name for p in self.mask_dir.glob('*.png'))

        if not self.files:
            raise RuntimeError(f"Aucun masque trouvé dans {self.mask_dir}")

    def __len__(self):
        return len(self.files)

    def _augment(self, rgb_pil, depth_np, mask_np):
        if random.random() < 0.5:
            rgb_pil = rgb_pil.transpose(Image.FLIP_LEFT_RIGHT)
            depth_np = np.fliplr(depth_np).copy()
            mask_np = np.fliplr(mask_np).copy()

        if random.random() < 0.5:
            w, h = rgb_pil.size
            scale = random.uniform(0.85, 1.0)
            cw, ch = int(w * scale), int(h * scale)
            x0 = random.randint(0, w - cw)
            y0 = random.randint(0, h - ch)
            rgb_pil = rgb_pil.crop((x0, y0, x0 + cw, y0 + ch))
            depth_np = depth_np[y0:y0 + ch, x0:x0 + cw]
            mask_np = mask_np[y0:y0 + ch, x0:x0 + cw]

        if random.random() < 0.5:
            rgb_pil = ImageEnhance.Brightness(rgb_pil).enhance(random.uniform(0.8, 1.2))
            rgb_pil = ImageEnhance.Contrast(rgb_pil).enhance(random.uniform(0.8, 1.2))
            rgb_pil = ImageEnhance.Color(rgb_pil).enhance(random.uniform(0.8, 1.2))

        return rgb_pil, depth_np, mask_np

    def __getitem__(self, idx):
        name = self.files[idx]
        rgb_pil = Image.open(self.rgb_dir / name).convert('RGB')
        depth_raw = np.array(Image.open(self.depth_dir / name))
        mask = np.array(Image.open(self.mask_dir / name))  # 0=sol, 1=obstacle, 2=inconnu

        if self.augment:
            rgb_pil, depth_raw, mask = self._augment(rgb_pil, depth_raw, mask)

        rgb = np.array(rgb_pil)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb = np.transpose(rgb, (2, 0, 1))

        depth = depth_raw.astype(np.float32) / 1000.0
        depth = np.nan_to_num(depth, nan=0.0, posinf=DEPTH_MAX_M, neginf=0.0)
        depth = np.clip(depth, 0.0, DEPTH_MAX_M)
        depth = cv2.resize(depth, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        depth = depth / DEPTH_MAX_M

        # nearest: préserve exactement 0/1/2, jamais de valeur intermédiaire
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        return (torch.from_numpy(rgb).float(),
                torch.from_numpy(depth).unsqueeze(0).float(),
                torch.from_numpy(mask).long())