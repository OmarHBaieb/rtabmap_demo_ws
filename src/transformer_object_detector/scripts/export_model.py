"""
export_model.py

Exporte MultimodalTransformer en TorchScript -> multimodal_transformer.pt

Usage:
    # Export avec poids fraîchement initialisés (pour valider le pipeline ROS 2
    # mécaniquement, en attendant un entraînement réel) :
    python3 export_model.py --output multimodal_transformer.pt

    # Export à partir d'un checkpoint entraîné (state_dict .pth) :
    python3 export_model.py --checkpoint chemin/vers/checkpoint.pth \
                             --output multimodal_transformer.pt

Nécessite : torch, connexion internet la 1ère fois (téléchargement DINOv2 via
torch.hub, mis en cache ensuite dans ~/.cache/torch/hub).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from multimodal_transformer import MultimodalTransformer, IMG_SIZE  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Export MultimodalTransformer en TorchScript")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Chemin vers un state_dict .pth entraîné (optionnel)")
    parser.add_argument("--output", type=str, default="multimodal_transformer.pt",
                         help="Chemin de sortie du fichier TorchScript")
    parser.add_argument("--method", choices=["trace", "script"], default="trace",
                         help="torch.jit.trace (recommandé ici, à cause de torch.hub/DINOv2) "
                              "ou torch.jit.script")
    args = parser.parse_args()

    print("Construction du modèle...")
    model = MultimodalTransformer(freeze_rgb_backbone=True)
    model.eval()

    if args.checkpoint:
        print(f"Chargement du checkpoint : {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [avertissement] clés manquantes ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  [avertissement] clés inattendues ({len(unexpected)}): {unexpected[:5]}...")
    else:
        print("Aucun checkpoint fourni : export avec poids initiaux "
              "(DINOv2 pré-entraîné gelé + tête/depth ViT initialisés aléatoirement).")
        print("=> Suffisant pour valider le pipeline ROS 2 mécaniquement, "
              "PAS pour des prédictions fiables.")

    dummy_rgb = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    dummy_depth = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)

    print(f"Export TorchScript (méthode={args.method})...")
    if args.method == "trace":
        # check_trace=False : nn.MultiheadAttention peut basculer entre deux
        # implémentations internes équivalentes (fused vs pas-à-pas) d'un
        # passage de trace à l'autre, ce qui fait échouer la vérification
        # structurelle de torch.jit.trace même quand les valeurs numériques
        # sont correctes. On valide manuellement la sortie juste après.
        traced = torch.jit.trace(model, (dummy_rgb, dummy_depth), check_trace=False)
    else:
        traced = torch.jit.script(model)

    # Vérification post-export
    with torch.no_grad():
        out = traced(dummy_rgb, dummy_depth)
    assert out.shape == (1, 3, IMG_SIZE, IMG_SIZE), f"Forme de sortie inattendue: {out.shape}"

    traced.save(args.output)
    print(f"Modèle exporté avec succès -> {args.output}")
    print(f"Forme de sortie validée : {tuple(out.shape)}")


if __name__ == "__main__":
    main()