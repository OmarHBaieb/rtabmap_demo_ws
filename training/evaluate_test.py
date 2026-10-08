import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "src" / "transformer_object_detector" / "models"))
from multimodal_transformer import MultimodalTransformer
from traversability_dataset import TraversabilityDataset
from losses import IGNORE_INDEX

NUM_CLASSES = 2


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total_px = 0, 0
    inter = torch.zeros(NUM_CLASSES)
    union = torch.zeros(NUM_CLASSES)
    for rgb, depth, mask in loader:
        rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)
        pred = torch.argmax(model(rgb, depth), dim=1)
        valid = mask != IGNORE_INDEX
        correct += ((pred == mask) & valid).sum().item()
        total_px += valid.sum().item()
        for c in range(NUM_CLASSES):
            p = (pred == c) & valid
            g = (mask == c) & valid
            inter[c] += (p & g).sum().item()
            union[c] += (p | g).sum().item()
    return correct / max(total_px, 1), inter / torch.clamp(union, min=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nyu_root', required=True)
    p.add_argument('--masks_root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    test_ds = TraversabilityDataset(args.nyu_root, args.masks_root, split='test', augment=False)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MultimodalTransformer(freeze_rgb_backbone=True).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))

    acc, iou = evaluate(model, loader, args.device)
    print(f"Pixel accuracy: {acc:.4f}")
    print(f"mIoU (sol+obstacle): {iou.mean().item():.4f}")
    print(f"IoU sol: {iou[0]:.4f} | obstacle: {iou[1]:.4f}")


if __name__ == '__main__':
    main()