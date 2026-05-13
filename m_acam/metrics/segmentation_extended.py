"""Segmentation metrics for paper tables: Dice, IoU, pixel P/R, boundary F1, HD95."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def dice_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    p = (pred > 0.5).float()
    t = (target > 0.5).float()
    inter = (p * t).sum()
    union_d = p.sum() + t.sum()
    union_i = p.sum() + t.sum() - inter
    dice = float((2.0 * inter + eps) / (union_d + eps))
    iou = float((inter + eps) / (union_i + eps))
    return dice, iou


def pixel_precision_recall(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    p = (pred > 0.5).bool().cpu().numpy().squeeze()
    t = (target > 0.5).bool().cpu().numpy().squeeze()
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, np.logical_not(t)).sum())
    fn = int(np.logical_and(np.logical_not(p), t).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return float(prec), float(rec)


def boundary_f1(pred: torch.Tensor, target: torch.Tensor, tolerance_px: int = 2) -> float:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    k = 2 * tolerance_px + 1
    pool = torch.nn.MaxPool2d(kernel_size=k, stride=1, padding=tolerance_px)
    pred_b = torch.clamp(pred - F.avg_pool2d(pred, 3, 1, 1), min=0)
    tgt_b = torch.clamp(target - F.avg_pool2d(target, 3, 1, 1), min=0)
    pred_d = pool(pred_b)
    tgt_d = pool(tgt_b)
    tp = (pred_b * tgt_d).sum()
    fp = (pred_b * (1 - tgt_d)).sum()
    fn = (tgt_b * (1 - pred_d)).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    bf1 = 2 * precision * recall / (precision + recall + 1e-6)
    return float(bf1)


def _boundary_coords(mask_bool: np.ndarray) -> np.ndarray:
    m = mask_bool.astype(bool)
    if not m.any():
        return np.zeros((0, 2), dtype=np.float64)
    er = binary_erosion(m)
    ys, xs = np.where(m & ~er)
    return np.column_stack((ys.astype(np.float64), xs.astype(np.float64)))


def hd95_pixels(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = (pred > 0.5).bool().cpu().numpy().squeeze()
    t = (target > 0.5).bool().cpu().numpy().squeeze()
    pa, ta = _boundary_coords(p), _boundary_coords(t)
    if pa.shape[0] == 0 and ta.shape[0] == 0:
        return 0.0
    if pa.shape[0] == 0 or ta.shape[0] == 0:
        return float("nan")

    def directed_h95(a: np.ndarray, b: np.ndarray) -> float:
        tree = cKDTree(b)
        d, _ = tree.query(a)
        return float(np.percentile(d, 95))

    d1 = directed_h95(pa, ta)
    d2 = directed_h95(ta, pa)
    return max(d1, d2)


def hd95_to_mm(hd_px: float, spacing_mm: float) -> float:
    if np.isnan(hd_px):
        return float("nan")
    return float(hd_px * spacing_mm)
