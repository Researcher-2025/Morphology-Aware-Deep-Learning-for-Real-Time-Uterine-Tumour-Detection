"""
PASCAL VOC–style AP for single-class detection (one GT box per image or empty GT).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO/VOC style AP: integral of precision envelope over recall."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = float(np.maximum(mpre[i - 1], mpre[i]))
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def compute_ap_voc(
    detections: Sequence[Tuple[int, float, np.ndarray]],
    gt_boxes_xyxy: Sequence[Optional[np.ndarray]],
    iou_threshold: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    detections: list of (image_index, score, pred_box_xyxy)
    gt_boxes_xyxy: len = num_images, None if no object (negative image)
    Returns AP and the precision/recall arrays used (after sorting detections).
    """
    num_images = len(gt_boxes_xyxy)
    n_gt = sum(1 for g in gt_boxes_xyxy if g is not None)
    if n_gt == 0:
        return 0.0, np.array([]), np.array([])

    dets = sorted(detections, key=lambda x: -x[1])
    gt_matched = np.zeros(num_images, dtype=bool)

    tp_list: List[float] = []
    fp_list: List[float] = []

    for img_i, _, pred_box in dets:
        g = gt_boxes_xyxy[img_i]
        if g is None:
            tp_list.append(0.0)
            fp_list.append(1.0)
            continue
        iou = box_iou_np(pred_box, g)
        if iou >= iou_threshold and not gt_matched[img_i]:
            tp_list.append(1.0)
            fp_list.append(0.0)
            gt_matched[img_i] = True
        else:
            tp_list.append(0.0)
            fp_list.append(1.0)

    tp_c = np.cumsum(tp_list)
    fp_c = np.cumsum(fp_list)
    precisions = tp_c / np.maximum(tp_c + fp_c, 1.0)
    recalls = tp_c / float(n_gt)
    ap = ap_from_pr(recalls, precisions)
    return ap, precisions, recalls


def box_iou_np(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return float(inter / (area_a + area_b - inter + 1e-8))


def operating_point_f1(
    gt_boxes_xyxy: Sequence[Optional[np.ndarray]],
    pred_boxes_per_image: Sequence[np.ndarray],
    pred_scores_per_image: Sequence[np.ndarray],
    score_thresh: float,
    match_iou: float,
) -> Tuple[float, float, float, int, int, int]:
    """Per-image top matching: best prediction above score_thresh vs single GT."""
    tp = fp = fn = 0
    for g, boxes, scores in zip(gt_boxes_xyxy, pred_boxes_per_image, pred_scores_per_image):
        if g is None:
            if len(scores) == 0:
                continue
            keep = scores >= score_thresh
            if keep.any():
                fp += int(np.count_nonzero(keep))
            continue

        if len(boxes) == 0 or len(scores) == 0:
            fn += 1
            continue

        best_j = int(np.argmax(scores))
        if float(scores[best_j]) < score_thresh:
            fn += 1
            continue

        iou = box_iou_np(boxes[best_j], g)
        if iou >= match_iou:
            tp += 1
            fp += int(np.count_nonzero(scores >= score_thresh) - 1)
        else:
            fp += int(np.count_nonzero(scores >= score_thresh))
            fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1, tp, fp, fn
