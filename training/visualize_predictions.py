import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "src" / "transformer_object_detector" / "models"))
from multimodal_transformer import MultimodalTransformer
from traversability_dataset import TraversabilityDataset
from losses import IGNORE_INDEX

# 0=sol(vert), 1=obstacle(rouge), 2=inconnu(gris, affiché seulement en vérité terrain)
CLASS_COLORS = np.array([[0, 200, 0], [220, 0, 0], [128, 128, 128]], dtype=np.uint8)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize_rgb(rgb_tensor):
    img = rgb_tensor.permute(1, 2, 0).numpy()
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nyu_root', required=True)
    p.add_argument('--masks_root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--num_samples', type=int, default=8)
    p.add_argument('--output_dir', default='predictions_preview')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    model = MultimodalTransformer(freeze_rgb_backbone=True).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    model.eval()

    dataset = TraversabilityDataset(args.nyu_root, args.masks_root, split='test')
    indices = np.random.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            rgb, depth, mask_gt = dataset[idx]
            logits = model(rgb.unsqueeze(0).to(args.device), depth.unsqueeze(0).to(args.device))
            # seuls les 2 premiers canaux sont entraînés (voir detector_node.py)
            pred = torch.argmax(logits[:, :2], dim=1).squeeze(0).cpu().numpy()

            gt = mask_gt.numpy()
            gt_display = np.where(gt == IGNORE_INDEX, 2, gt)  # affichage seulement

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(denormalize_rgb(rgb)); axes[0].set_title("RGB")
            axes[1].imshow(depth.squeeze(0).numpy(), cmap='viridis'); axes[1].set_title("Depth")
            axes[2].imshow(CLASS_COLORS[gt_display]); axes[2].set_title("Vérité terrain (gris=inconnu, ignoré)")
            axes[3].imshow(CLASS_COLORS[pred]); axes[3].set_title("Prédiction (sol/obstacle uniquement)")
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(out_dir / f"sample_{i:02d}_idx{idx}.png", dpi=100)
            plt.close(fig)

    print(f"Images sauvegardées dans {out_dir}/ — vert=sol, rouge=obstacle, gris=inconnu(ignoré)")


if __name__ == '__main__':
    main()