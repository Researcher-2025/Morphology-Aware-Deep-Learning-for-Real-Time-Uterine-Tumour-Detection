from typing import Dict, List, Tuple

import torch

from m_acam.model import non_max_suppression


def box_iou_xyxy(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    x1 = torch.maximum(box_a[0], box_b[0])
    y1 = torch.maximum(box_a[1], box_b[1])
    x2 = torch.minimum(box_a[2], box_b[2])
    y2 = torch.minimum(box_a[3], box_b[3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_a = torch.clamp(box_a[2] - box_a[0], min=0) * torch.clamp(box_a[3] - box_a[1], min=0)
    area_b = torch.clamp(box_b[2] - box_b[0], min=0) * torch.clamp(box_b[3] - box_b[1], min=0)
    return float(inter / (area_a + area_b - inter + 1e-8))


def decode_scale(pred: torch.Tensor, anchors: torch.Tensor, image_size: int, conf_thresh: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # pred: [A, H, W, 6] for single class
    a, h, w, _ = pred.shape
    device = pred.device
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    xx = xx.unsqueeze(0).expand(a, -1, -1).float()
    yy = yy.unsqueeze(0).expand(a, -1, -1).float()

    tx = torch.sigmoid(pred[..., 0])
    ty = torch.sigmoid(pred[..., 1])
    tw = pred[..., 2]
    th = pred[..., 3]
    obj = torch.sigmoid(pred[..., 4])
    cls = torch.sigmoid(pred[..., 5])
    conf = obj * cls

    cx = (xx + tx) / w * image_size
    cy = (yy + ty) / h * image_size
    aw = anchors[:, 0].view(a, 1, 1)
    ah = anchors[:, 1].view(a, 1, 1)
    bw = torch.exp(tw).clamp(max=1e3) * aw
    bh = torch.exp(th).clamp(max=1e3) * ah

    x1 = cx - bw * 0.5
    y1 = cy - bh * 0.5
    x2 = cx + bw * 0.5
    y2 = cy + bh * 0.5
    boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)
    scores = conf.reshape(-1)

    keep = scores >= conf_thresh
    return boxes[keep], scores[keep]


def decode_detections(
    outputs: List[torch.Tensor],
    anchors: Dict[str, torch.Tensor],
    conf_thresh: float = 0.25,
    nms_iou: float = 0.45,
    image_size: int = 512,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    scale_names = ["p3", "p4", "p5"]
    batch_size = outputs[0].shape[0]
    decoded = []
    for bi in range(batch_size):
        boxes_all = []
        scores_all = []
        for si, pred in enumerate(outputs):
            _, _, h, w = pred.shape
            p = pred[bi].view(3, 6, h, w).permute(0, 2, 3, 1).contiguous()
            boxes, scores = decode_scale(p, anchors[scale_names[si]].to(pred.device), image_size=image_size, conf_thresh=conf_thresh)
            boxes_all.append(boxes)
            scores_all.append(scores)
        boxes_cat = torch.cat(boxes_all, dim=0) if boxes_all else torch.empty((0, 4))
        scores_cat = torch.cat(scores_all, dim=0) if scores_all else torch.empty((0,))
        if boxes_cat.numel() == 0:
            decoded.append((boxes_cat, scores_cat))
            continue
        keep = non_max_suppression(boxes_cat, scores_cat, iou_threshold=nms_iou)
        decoded.append((boxes_cat[keep], scores_cat[keep]))
    return decoded


def average_precision_11_point(pred_rows: List[Tuple[float, int]], total_gt: int) -> float:
    if total_gt <= 0:
        return 0.0
    if not pred_rows:
        return 0.0
    pred_rows = sorted(pred_rows, key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    for _, is_tp in pred_rows:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / max(total_gt, 1))
    ap = 0.0
    for thr in [i / 10.0 for i in range(11)]:
        pmax = 0.0
        for p, r in zip(precisions, recalls):
            if r >= thr:
                pmax = max(pmax, p)
        ap += pmax
    return ap / 11.0
