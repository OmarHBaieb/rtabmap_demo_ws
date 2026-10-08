import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

GROUND_CLASSES = {2, 20}      # floor, floor mat -> 0 (sol)
UNKNOWN_CLASSES = {0, 255}    # non labellisé + bordure invalide -> 2 (inconnu, ignoré)


def remap(seg40: np.ndarray) -> np.ndarray:
    out = np.full_like(seg40, fill_value=1, dtype=np.uint8)  # défaut: obstacle
    for c in GROUND_CLASSES:
        out[seg40 == c] = 0
    for c in UNKNOWN_CLASSES:
        out[seg40 == c] = 2
    return out


def process_split(nyu_root, output_root, split):
    seg_dir = nyu_root / 'seg40' / split
    if not seg_dir.exists():
        return 0
    out_dir = output_root / 'masks' / split
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(seg_dir.glob('*.png'))
    for f in tqdm(files, desc=split):
        mask = remap(np.array(Image.open(f)))
        Image.fromarray(mask, mode='L').save(out_dir / f.name)
    return len(files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nyu_root', required=True)
    p.add_argument('--output_root', required=True)
    args = p.parse_args()

    nyu_root = Path(args.nyu_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    total = sum(process_split(nyu_root, output_root, s) for s in ['train', 'test'])
    print(f"{total} masques générés dans {output_root / 'masks'} (0=sol, 1=obstacle, 2=inconnu/ignoré)")


if __name__ == '__main__':
    main()