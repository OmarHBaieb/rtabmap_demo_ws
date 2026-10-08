"""
train.py — sol(0)/obstacle(1) entraînés, unknown(255)=ignore_index partout
(loss, poids de classes, métriques, sélection de checkpoint).
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "src" / "transformer_object_detector" / "models"))
from multimodal_transformer import MultimodalTransformer
from traversability_dataset import TraversabilityDataset
from losses import CombinedLoss, IGNORE_INDEX

NUM_CLASSES = 2  # sol, obstacle — unknown n'est jamais une classe de sortie réelle


def make_train_val_split(nyu_root, masks_root, val_fraction=0.1, seed=42):
    full = TraversabilityDataset(nyu_root, masks_root, split='train', augment=False)
    files = list(full.files)
    random.Random(seed).shuffle(files)
    n_val = max(1, int(len(files) * val_fraction))
    return files[n_val:], files[:n_val]


def compute_class_weights(dataset, max_samples=200):
    counts = torch.zeros(NUM_CLASSES)
    n = min(len(dataset), max_samples)
    for i in range(n):
        _, _, mask = dataset[i]
        for c in range(NUM_CLASSES):
            counts[c] += (mask == c).sum()
    counts = torch.clamp(counts, min=1)
    return counts.sum() / (NUM_CLASSES * counts)


def compute_sample_weights(dataset, floor_boost=3.0):
    weights = []
    for name in dataset.files:
        mask = np.array(Image.open(dataset.mask_dir / name))
        valid = mask != IGNORE_INDEX
        frac = float((mask[valid] == 0).mean()) if valid.any() else 0.0
        weights.append(1.0 + frac * (floor_boost - 1.0))
    return torch.tensor(weights, dtype=torch.double)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    model.rgb_backbone.eval()
    total = 0.0
    for rgb, depth, mask in loader:
        rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu', enabled=(device == 'cuda')):
            logits = model(rgb, depth)
            loss = criterion(logits, mask)
        if device == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * rgb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total_px = 0.0, 0, 0
    inter = torch.zeros(NUM_CLASSES)
    union = torch.zeros(NUM_CLASSES)

    for rgb, depth, mask in loader:
        rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)
        logits = model(rgb, depth)
        total_loss += criterion(logits, mask).item() * rgb.size(0)

        valid = mask != IGNORE_INDEX
        pred = torch.argmax(logits, dim=1)
        correct += ((pred == mask) & valid).sum().item()
        total_px += valid.sum().item()

        for c in range(NUM_CLASSES):
            p = (pred == c) & valid
            g = (mask == c) & valid
            inter[c] += (p & g).sum().item()
            union[c] += (p | g).sum().item()

    iou = inter / torch.clamp(union, min=1)
    return total_loss / len(loader.dataset), correct / max(total_px, 1), iou


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nyu_root', required=True)
    p.add_argument('--masks_root', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='checkpoints/traversability_best.pth')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--log_dir', default='runs')
    p.add_argument('--ce_weight', type=float, default=0.4)
    p.add_argument('--dice_weight', type=float, default=0.6)
    p.add_argument('--floor_boost', type=float, default=3.0)
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"Device: {args.device} | Classes: 0=sol 1=obstacle | 255=unknown (ignoré)")

    train_files, val_files = make_train_val_split(args.nyu_root, args.masks_root, args.val_fraction)
    print(f"Split: {len(train_files)} train / {len(val_files)} val (test réservé à l'évaluation finale)")

    train_ds = TraversabilityDataset(args.nyu_root, args.masks_root, split='train',
                                      file_list=train_files, augment=True)
    val_ds = TraversabilityDataset(args.nyu_root, args.masks_root, split='train',
                                    file_list=val_files, augment=False)

    if args.floor_boost > 1.0:
        sw = compute_sample_weights(train_ds, args.floor_boost)
        sampler = WeightedRandomSampler(sw, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                   num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MultimodalTransformer(freeze_rgb_backbone=True).to(args.device)
    class_weights = compute_class_weights(train_ds).to(args.device)
    print(f"Poids classes [sol, obstacle]: {class_weights.tolist()}")

    criterion = CombinedLoss(class_weights=class_weights, num_classes=NUM_CLASSES,
                              ce_weight=args.ce_weight, dice_weight=args.dice_weight)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device == 'cuda'))

    best_iou, no_improve = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device, scaler)
        val_loss, acc, iou = evaluate(model, val_loader, criterion, args.device)
        scheduler.step()
        miou = iou.mean().item()

        print(f"[{epoch:03d}/{args.epochs}] train={tr_loss:.4f} val={val_loss:.4f} "
              f"acc={acc:.4f} mIoU={miou:.4f} (sol={iou[0]:.3f} obstacle={iou[1]:.3f}) [{time.time()-t0:.1f}s]")

        writer.add_scalars('Loss', {'train': tr_loss, 'val': val_loss}, epoch)
        writer.add_scalars('IoU', {'sol': iou[0].item(), 'obstacle': iou[1].item(), 'mIoU': miou}, epoch)

        if miou > best_iou:
            best_iou, no_improve = miou, 0
            torch.save(model.state_dict(), args.output)
            print(f"  -> checkpoint sauvegardé: {args.output}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping (patience={args.patience}).")
                break

    writer.close()
    print(f"\nMeilleur mIoU val: {best_iou:.4f} | Checkpoint: {args.output}")


if __name__ == '__main__':
    main()