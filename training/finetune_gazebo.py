"""
Fine-tuning ciblé : dégèle les N derniers blocs de DINOv2, entraîne sur les
frames Gazebo bootstrap-labellisées MÉLANGÉES avec un échantillon NYUv2
(rehearsal anti-oubli), valide à chaque epoch sur Gazebo ET sur le test
NYUv2 complet pour détecter toute régression.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "src" / "transformer_object_detector" / "models"))
from multimodal_transformer import MultimodalTransformer
from traversability_dataset import TraversabilityDataset
from losses import CombinedLoss, IGNORE_INDEX
from gazebo_dataset import GazeboDataset
from train import evaluate  # réutilise EXACTEMENT la fonction de validation existante

NUM_CLASSES = 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help="checkpoint pré-entraîné NYUv2 (.pth)")
    p.add_argument('--gazebo_frames', default='../gazebo_frames')
    p.add_argument('--gazebo_masks', default='gazebo_masks')
    p.add_argument('--nyu_root', required=True)
    p.add_argument('--nyu_masks_root', required=True)
    p.add_argument('--unfreeze_last_n_blocks', type=int, default=2)
    p.add_argument('--nyu_replay_ratio', type=float, default=2.0,
                   help="ratio NYUv2/Gazebo melange dans le training pour eviter l'oubli")
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--val_fraction', type=float, default=0.15)
    p.add_argument('--nyu_regression_tol', type=float, default=0.05,
                   help="mIoU NYUv2 ne doit pas chuter de plus que cette valeur vs baseline")
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='checkpoints/traversability_gazebo_finetuned.pth')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    device = args.device

    # --- modele : charge le checkpoint existant, degele les derniers blocs ---
    model = MultimodalTransformer(freeze_rgb_backbone=True,
                                   unfreeze_last_n_blocks=args.unfreeze_last_n_blocks).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Checkpoint charge: {args.checkpoint}")

    n_trainable = sum(p_.numel() for p_ in model.trainable_parameters())
    print(f"Parametres entrainables (dont {args.unfreeze_last_n_blocks} derniers blocs DINOv2): {n_trainable:,}")

    # --- split Gazebo train/val ---
    all_names = sorted(p_.stem for p_ in Path(args.gazebo_masks).glob("*.png"))
    random.Random(args.seed).shuffle(all_names)
    n_val = max(1, int(len(all_names) * args.val_fraction))
    val_names, train_names = all_names[:n_val], all_names[n_val:]
    print(f"Split Gazebo: {len(train_names)} train / {len(val_names)} val")

    gz_train_ds = GazeboDataset(args.gazebo_frames, args.gazebo_masks, file_list=train_names)
    val_ds = GazeboDataset(args.gazebo_frames, args.gazebo_masks, file_list=val_names)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- set NYUv2 test complet, pour la garde anti-regression (jamais dans le train) ---
    nyu_test_ds = TraversabilityDataset(args.nyu_root, args.nyu_masks_root, split='test', augment=False)
    nyu_test_loader = DataLoader(nyu_test_ds, batch_size=8, shuffle=False, num_workers=2)

    # --- set NYUv2 TRAIN, sous-echantillonne, melange au training Gazebo pour ---
    # --- eviter le catastrophic forgetting (anti-oubli par rehearsal) ---
    nyu_train_full = TraversabilityDataset(args.nyu_root, args.nyu_masks_root, split='train', augment=True)
    n_nyu_replay = min(len(nyu_train_full), int(len(gz_train_ds) * args.nyu_replay_ratio))
    replay_indices = random.Random(args.seed).sample(range(len(nyu_train_full)), n_nyu_replay)
    nyu_replay_ds = Subset(nyu_train_full, replay_indices)

    combined_train_ds = ConcatDataset([gz_train_ds, nyu_replay_ds])
    train_loader = DataLoader(combined_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Training combine: {len(gz_train_ds)} Gazebo + {len(nyu_replay_ds)} NYUv2 (replay) = {len(combined_train_ds)} total")

    criterion = CombinedLoss(num_classes=NUM_CLASSES, ce_weight=0.4, dice_weight=0.6)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=1e-4)

    # --- baseline NYUv2 AVANT tout fine-tuning, pour comparaison ---
    print("Evaluation baseline NYUv2 (avant fine-tuning)...")
    _, _, nyu_iou_baseline = evaluate(model, nyu_test_loader, criterion, device)
    nyu_miou_baseline = nyu_iou_baseline.mean().item()
    print(f"Baseline NYUv2 mIoU: {nyu_miou_baseline:.4f} (sol={nyu_iou_baseline[0]:.3f} obstacle={nyu_iou_baseline[1]:.3f})")

    best_gazebo_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        model.depth_backbone.train()
        if args.unfreeze_last_n_blocks > 0:
            model.rgb_backbone.train()
        else:
            model.rgb_backbone.eval()

        total_loss = 0.0
        for rgb, depth, mask in train_loader:
            rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(rgb, depth)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * rgb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        gz_val_loss, gz_acc, gz_iou = evaluate(model, val_loader, criterion, device)
        gz_miou = gz_iou.mean().item()

        nyu_val_loss, nyu_acc, nyu_iou = evaluate(model, nyu_test_loader, criterion, device)
        nyu_miou = nyu_iou.mean().item()
        nyu_drop = nyu_miou_baseline - nyu_miou

        print(f"[{epoch:02d}/{args.epochs}] train={train_loss:.4f} | "
              f"Gazebo: val={gz_val_loss:.4f} mIoU={gz_miou:.4f} (sol={gz_iou[0]:.3f} obs={gz_iou[1]:.3f}) | "
              f"NYUv2: mIoU={nyu_miou:.4f} (drop={nyu_drop:+.4f}) [{time.time()-t0:.1f}s]")

        if nyu_drop > args.nyu_regression_tol:
            print(f"REGRESSION NYUv2 detectee (drop={nyu_drop:.4f} > tol={args.nyu_regression_tol}). Arret.")
            break

        if gz_miou > best_gazebo_iou:
            best_gazebo_iou = gz_miou
            torch.save(model.state_dict(), args.output)
            print(f"  -> checkpoint sauvegarde: {args.output}")

    print(f"\nMeilleur mIoU Gazebo: {best_gazebo_iou:.4f} | Checkpoint: {args.output}")
    print(f"Baseline NYUv2 mIoU: {nyu_miou_baseline:.4f}")


if __name__ == '__main__':
    main()
