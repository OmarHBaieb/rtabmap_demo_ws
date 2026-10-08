"""
multimodal_transformer.py

Architecture du modèle de segmentation RGB-D pour rtabmap_demo_ws.

Composants :
  - Backbone RGB   : DINOv2-S/14 gelé (chargé via torch.hub), features denses
  - Backbone depth : ViT entraîné from scratch (patch 14, dim 384, 6 couches, 6 têtes)
  - Fusion         : cross-attention (Q = tokens RGB, K/V = tokens depth)
  - Tête           : segmentation 3 classes (0=sol, 1=obstacle, 2=inconnu)
                      sortie [B, 3, 224, 224]

Ce fichier ne dépend QUE de torch/torchvision (+ torch.hub pour DINOv2).
Il doit rester import-safe pour TorchScript (pas de branches Python
dynamiques dépendant de valeurs de tenseurs dans forward()).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


IMG_SIZE = 224
PATCH_SIZE = 14
EMBED_DIM = 384          # dim DINOv2-S et du ViT depth (doivent matcher pour la cross-attn)
NUM_HEADS = 6
DEPTH_LAYERS = 6
NUM_CLASSES = 3
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 16*16 = 256


# ---------------------------------------------------------------------------
# Backbone depth : ViT from scratch
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Découpe l'image depth en patches 14x14 et projette en embed_dim."""

    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE,
                 in_chans=1, embed_dim=EMBED_DIM):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim=EMBED_DIM, num_heads=NUM_HEADS, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DepthViT(nn.Module):
    """ViT depth from scratch : patch 14, dim 384, 6 couches, 6 têtes."""

    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE,
                 embed_dim=EMBED_DIM, depth=DEPTH_LAYERS, num_heads=NUM_HEADS):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans=1, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(depth)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Backbone RGB : DINOv2-S/14, gel total ou partiel selon les paramètres
# ---------------------------------------------------------------------------
class DinoV2RGBBackbone(nn.Module):
    """
    Charge DINOv2-S/14 via torch.hub.
    freeze=True, unfreeze_last_n_blocks=0        -> comportement original : tout gelé
    freeze=True, unfreeze_last_n_blocks=N (N>0)  -> dégèle les N derniers blocs + norm finale
    freeze=False                                  -> tout dégelé

    NOTE: nécessite une connexion internet la première fois (téléchargement
    des poids depuis facebookresearch/dinov2 via torch.hub), sauf si un
    cache local est déjà présent (~/.cache/torch/hub).
    """

    def __init__(self, freeze: bool = True, unfreeze_last_n_blocks: int = 0):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.embed_dim = self.backbone.embed_dim  # 384 pour dinov2_vits14

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.any_trainable = False
        if not freeze:
            for p in self.backbone.parameters():
                p.requires_grad = True
            self.any_trainable = True
        elif unfreeze_last_n_blocks > 0:
            n_blocks = len(self.backbone.blocks)
            assert unfreeze_last_n_blocks <= n_blocks, \
                f"unfreeze_last_n_blocks={unfreeze_last_n_blocks} > {n_blocks} blocs disponibles"
            for blk in self.backbone.blocks[n_blocks - unfreeze_last_n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True
            self.any_trainable = True

        if not self.any_trainable:
            self.backbone.eval()

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.any_trainable:
            feats = self.backbone.forward_features(rgb)
        else:
            with torch.no_grad():
                feats = self.backbone.forward_features(rgb)
        patch_tokens = feats['x_norm_patchtokens']  # [B, 256, 384]
        return patch_tokens


# ---------------------------------------------------------------------------
# Fusion cross-attention : Q = RGB, K/V = Depth
# ---------------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=EMBED_DIM, num_heads=NUM_HEADS, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, rgb_tokens: torch.Tensor, depth_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(rgb_tokens)
        kv = self.norm_kv(depth_tokens)
        fused, _ = self.cross_attn(q, kv, kv, need_weights=False)
        fused = rgb_tokens + fused
        fused = fused + self.mlp(self.norm_out(fused))
        return fused  # [B, 256, 384]


# ---------------------------------------------------------------------------
# Tête de segmentation
# ---------------------------------------------------------------------------
class SegmentationHead(nn.Module):
    """
    Reprojette les tokens fusionnés [B, 256, 384] en grille spatiale [B, 384, 16, 16]
    puis upsample par convolutions transposées jusqu'à 224x224, 3 classes.
    """

    def __init__(self, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES,
                 grid_size=IMG_SIZE // PATCH_SIZE):
        super().__init__()
        self.grid_size = grid_size  # 16
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),          # 32->64
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),           # 64->128
            nn.GELU(),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = fused_tokens.shape
        x = fused_tokens.transpose(1, 2).reshape(b, c, self.grid_size, self.grid_size)
        x = self.decoder(x)  # [B, 3, 128, 128]
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        return x  # [B, 3, 224, 224]


# ---------------------------------------------------------------------------
# Modèle complet
# ---------------------------------------------------------------------------
class MultimodalTransformer(nn.Module):
    def __init__(self, freeze_rgb_backbone: bool = True, unfreeze_last_n_blocks: int = 0):
        super().__init__()
        self.rgb_backbone = DinoV2RGBBackbone(freeze=freeze_rgb_backbone,
                                               unfreeze_last_n_blocks=unfreeze_last_n_blocks)
        self.depth_backbone = DepthViT()
        self.fusion = CrossAttentionFusion(dim=self.rgb_backbone.embed_dim)
        self.head = SegmentationHead(embed_dim=self.rgb_backbone.embed_dim)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        rgb   : [B, 3, 224, 224] normalisé (ImageNet mean/std)
        depth : [B, 1, 224, 224] en mètres, clip [0, 3] puis normalisé /3.0
        return: logits [B, 3, 224, 224]
        """
        rgb_tokens = self.rgb_backbone(rgb)
        depth_tokens = self.depth_backbone(depth)
        fused = self.fusion(rgb_tokens, depth_tokens)
        logits = self.head(fused)
        return logits

    def trainable_parameters(self):
        """Retourne uniquement les paramètres entraînables (exclut DINOv2 gelé)."""
        return [p for p in self.parameters() if p.requires_grad]


if __name__ == "__main__":
    model = MultimodalTransformer()
    model.eval()
    rgb = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    depth = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = model(rgb, depth)
    print("Output shape:", out.shape)  # attendu: [1, 3, 224, 224]
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"Paramètres entraînables (hors DINOv2 gelé): {n_trainable:,}")
