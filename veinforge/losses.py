"""Segmentation losses for the P2 DL trainer (torch).

Includes clDice (centerline Dice, Shit et al. 2021) which rewards *connectivity*
of thin tubular structures like veins — so predicted networks don't break apart.
Imported only when torch is available (the [dl] extra).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Differentiable approximate skeleton (iterative morphological thinning)."""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cl_dice_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                 iters: int = 5, smooth: float = 1.0) -> torch.Tensor:
    """1 - centerline Dice between sigmoid(pred_logits) and target (B,1,H,W)."""
    pred = torch.sigmoid(pred_logits)
    skel_pred = soft_skel(pred, iters)
    skel_true = soft_skel(target, iters)
    tprec = (torch.sum(skel_pred * target) + smooth) / (torch.sum(skel_pred) + smooth)
    tsens = (torch.sum(skel_true * pred) + smooth) / (torch.sum(skel_true) + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dice = 1.0 - (2.0 * (prob * target).sum() + 1.0) / (prob.sum() + target.sum() + 1.0)
    return bce + dice


def dice_bce_cldice_loss(logits: torch.Tensor, target: torch.Tensor,
                         cldice_weight: float = 0.5) -> torch.Tensor:
    """BCE + Dice + clDice (connectivity-aware) combined loss."""
    return dice_bce_loss(logits, target) + cldice_weight * cl_dice_loss(logits, target)
