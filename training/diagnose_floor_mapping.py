"""
diagnose_floor_mapping.py

Deux vérifications sur un échantillon de test :
1. Quelles classes NYU40 réelles (valeurs brutes, pas couleurs) composent
   l'image, et quelle proportion est "floor"/"floor mat".
2. Overlay semi-transparent du masque sur le RGB, pour repérer visuellement
   un éventuel désalignement (le masque ne correspondrait pas aux vrais
   contours des objets dans la photo).

Usage:
    python3 diagnose_floor_mapping.py --nyu_root ~/nyu_data/NYUv2 \
        --masks_root ~/nyu_data/traversability --split test --num_samples 6
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

NYU40_NAMES = [
    "unlabeled", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "blinds", "desk",
    "shelves", "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes",
    "ceiling", "books", "refrigerator", "television", "paper", "towel",
    "shower curtain", "box", "whiteboard", "person", "night stand", "toilet",
    "sink", "lamp", "bathtub", "bag", "otherstructure", "otherfurniture", "otherprop"
]
GROUND_CLASSES = {2, 20}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nyu_root', required=True)
    p.add_argument('--masks_root', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--num_samples', type=int, default=6)
    p.add_argument('--output_dir', default='diagnose_preview')
    args = p.parse_args()

    nyu_root = Path(args.nyu_root).expanduser()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    seg_dir = nyu_root / 'seg40' / args.split
    rgb_dir = nyu_root / 'image' / args.split
    files = sorted(seg_dir.glob('*.png'))
    rng = np.random.default_rng(0)
    sample_files = rng.choice(files, size=min(args.num_samples, len(files)), replace=False)

    for f in sample_files:
        seg40 = np.array(Image.open(f))
        rgb = np.array(Image.open(rgb_dir / f.name).convert('RGB'))

        classes, counts = np.unique(seg40, return_counts=True)
        total = seg40.size
        floor_pct = sum(n for c, n in zip(classes, counts) if c in GROUND_CLASSES) / total * 100

        print(f"\n=== {f.name} ===")
        for c, n in sorted(zip(classes, counts), key=lambda x: -x[1])[:6]:
            name = NYU40_NAMES[c] if c < len(NYU40_NAMES) else f"classe_{c}"
            marker = " <- SOL" if c in GROUND_CLASSES else ""
            print(f"  {name:15s}: {100*n/total:5.1f}%{marker}")
        print(f"  Total sol réel (NYU40): {floor_pct:.1f}%")

        floor_mask = np.isin(seg40, list(GROUND_CLASSES))
        overlay = rgb.copy()
        overlay[floor_mask] = (overlay[floor_mask] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgb); axes[0].set_title("RGB original")
        axes[1].imshow(overlay); axes[1].set_title(f"Sol NYU40 réel en rouge ({floor_pct:.1f}%)")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / f"diag_{f.name}", dpi=100)
        plt.close(fig)

    print(f"\nImages sauvegardées dans {out_dir}/")
    print("Vérifie: le rouge (sol réel NYU40) correspond-il visuellement au vrai sol dans l'image ?")


if __name__ == '__main__':
    main()
    