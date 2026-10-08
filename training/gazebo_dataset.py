"""
Dataset pour les frames Gazebo capturées + masques bootstrap géométriques.
Réutilise exactement le preprocessing de detector_node.py (source unique de vérité)
pour garantir zéro divergence entre training et inférence.
"""
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "src" / "transformer_object_detector" / "transformer_object_detector"))
from detector_node import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, DEPTH_MAX_M


class GazeboDataset(Dataset):
    def __init__(self, frames_dir, masks_dir, file_list=None):
        self.frames_dir = Path(frames_dir)
        self.masks_dir = Path(masks_dir)
        if file_list is None:
            file_list = sorted(p.stem for p in self.masks_dir.glob("*.png"))
        self.names = file_list

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        rgb_bgr = np.load(self.frames_dir / f"{name}_rgb.npy")   # BGR, uint8
        depth_raw = np.load(self.frames_dir / f"{name}_depth.npy")  # float32, mètres
        mask = cv2.imread(str(self.masks_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)

        # --- RGB: même logique que preprocess_rgb de detector_node.py ---
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb = np.transpose(rgb, (2, 0, 1))

        # --- Depth: même logique que preprocess_depth de detector_node.py ---
        depth = depth_raw.astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=DEPTH_MAX_M, neginf=0.0)
        depth[depth > 50.0] = DEPTH_MAX_M
        depth = np.clip(depth, 0.0, DEPTH_MAX_M)
        depth = cv2.resize(depth, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        depth = depth / DEPTH_MAX_M

        # --- Mask: nearest pour préserver 0/1/2 exactement ---
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        return (torch.from_numpy(rgb).float(),
                torch.from_numpy(depth).unsqueeze(0).float(),
                torch.from_numpy(mask).long())