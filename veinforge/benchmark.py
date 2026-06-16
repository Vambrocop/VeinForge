"""Segmentation benchmark metrics — compare any segmenter against ground-truth masks."""
from __future__ import annotations
import numpy as np


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else 1.0


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    return float(2.0 * np.logical_and(pred, gt).sum() / denom) if denom else 1.0


def segmentation_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {"iou": iou(pred, gt), "dice": dice(pred, gt)}
