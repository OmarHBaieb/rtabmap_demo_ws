import torch
import torch.nn as nn
import torch.nn.functional as F

# Valeur utilisée dans les fichiers masque sur disque (0=sol, 1=obstacle, 2=inconnu)
IGNORE_INDEX = 2

# PyTorch exige que ignore_index soit hors de la plage couverte par `weight`
# quand weight est fourni (sinon: "weight tensor should be defined either for
# all or no classes"). On retranspose donc 2 -> 255 uniquement en interne,
# juste avant d'appeler CrossEntropyLoss/Dice — les fichiers sur disque et le
# reste du code continuent d'utiliser 2.
_CE_IGNORE_INDEX = 255


class DiceLoss(nn.Module):
    def __init__(self, num_classes=2, smooth=1e-6, class_weights=None, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        valid = (target != self.ignore_index)
        target_safe = target.clone()
        target_safe[~valid] = 0  # valeur bidon, masquée juste après

        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target_safe, self.num_classes).permute(0, 3, 1, 2).float()
        mask = valid.unsqueeze(1).float()

        probs = probs * mask
        target_onehot = target_onehot * mask

        dims = (0, 2, 3)
        inter = (probs * target_onehot).sum(dims)
        union = probs.sum(dims) + target_onehot.sum(dims)
        dice_per_class = (2 * inter + self.smooth) / (union + self.smooth)

        if self.class_weights is not None:
            w = self.class_weights / self.class_weights.sum()
            return 1 - (dice_per_class * w).sum()
        return 1 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, num_classes=2, ce_weight=0.4, dice_weight=0.6,
                 ignore_index=IGNORE_INDEX):
        super().__init__()
        # CrossEntropyLoss: remap l'ignore_index à 255 en interne (voir note
        # plus haut) — indépendant de la valeur `ignore_index` passée, qui
        # reste celle utilisée sur les fichiers/masques (2 par défaut).
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=_CE_IGNORE_INDEX)
        self._mask_ignore_index = ignore_index
        self.dice = DiceLoss(num_classes=num_classes, class_weights=class_weights,
                              ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        # Le modèle sort toujours 3 canaux (architecture inchangée), mais seuls
        # les 2 premiers (sol, obstacle) sont entraînés — le 3e reste inutilisé,
        # cohérent avec detector_node.py qui restreint aussi l'argmax à [:2].
        logits = logits[:, :2]
        # remap uniquement pour la CrossEntropy (Dice gère l'ignore_index elle-même)
        target_ce = torch.where(target == self._mask_ignore_index,
                                 torch.full_like(target, _CE_IGNORE_INDEX), target)
        return self.ce_weight * self.ce(logits, target_ce) + self.dice_weight * self.dice(logits, target)