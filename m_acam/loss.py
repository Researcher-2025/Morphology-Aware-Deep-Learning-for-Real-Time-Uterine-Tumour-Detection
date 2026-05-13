from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        anchors: Dict[str, torch.Tensor],
        num_classes: int = 1,
        image_size: int = 512,
        lambda_box: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_cls: float = 1.0,
    ) -> None:
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.image_size = image_size
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

    @staticmethod
    def _best_anchor(wh: torch.Tensor, anchors: torch.Tensor) -> int:
        inter = torch.min(wh[0], anchors[:, 0]) * torch.min(wh[1], anchors[:, 1])
        union = wh[0] * wh[1] + anchors[:, 0] * anchors[:, 1] - inter
        iou = inter / (union + 1e-8)
        return int(torch.argmax(iou).item())

    def forward(self, preds: List[torch.Tensor], targets: Dict) -> torch.Tensor:
        # Simplified YOLOv3 objective with single dominant box per image.
        total_loss = torch.tensor(0.0, device=preds[0].device)
        boxes_list = targets["boxes"]
        labels_list = targets["labels"]
        bsz = preds[0].shape[0]
        scales = ["p3", "p4", "p5"]

        for scale_idx, pred in enumerate(preds):
            b, c, h, w = pred.shape
            a = 3
            pred = pred.view(b, a, 5 + self.num_classes, h, w).permute(0, 1, 3, 4, 2).contiguous()
            obj_target = torch.zeros((b, a, h, w), device=pred.device)
            box_target = torch.zeros((b, a, h, w, 4), device=pred.device)
            cls_target = torch.zeros((b, a, h, w, self.num_classes), device=pred.device)
            box_mask = torch.zeros((b, a, h, w), device=pred.device)

            for bi in range(bsz):
                gt_box = boxes_list[bi][0].to(pred.device)
                if labels_list[bi][0].item() <= 0:
                    continue
                cxcywh = bbox_xyxy_to_cxcywh(gt_box.unsqueeze(0))[0]
                gx = (cxcywh[0] / self.image_size) * w
                gy = (cxcywh[1] / self.image_size) * h
                gw = cxcywh[2]
                gh = cxcywh[3]
                gi = int(torch.clamp(gx.floor(), min=0, max=w - 1).item())
                gj = int(torch.clamp(gy.floor(), min=0, max=h - 1).item())

                anchor_set = self.anchors[scales[scale_idx]].to(pred.device)
                best_a = self._best_anchor(torch.tensor([gw, gh], device=pred.device), anchor_set)
                obj_target[bi, best_a, gj, gi] = 1.0
                tx = gx - gi
                ty = gy - gj
                tw = torch.log((gw / (anchor_set[best_a, 0] + 1e-8)).clamp(min=1e-6))
                th = torch.log((gh / (anchor_set[best_a, 1] + 1e-8)).clamp(min=1e-6))
                box_target[bi, best_a, gj, gi] = torch.stack([tx, ty, tw, th])
                cls_target[bi, best_a, gj, gi, 0] = 1.0
                box_mask[bi, best_a, gj, gi] = 1.0

            pred_box = pred[..., :4]
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]

            loss_obj = F.binary_cross_entropy_with_logits(pred_obj, obj_target, reduction="mean")
            if box_mask.sum() > 0:
                loss_box = F.smooth_l1_loss(pred_box[box_mask > 0], box_target[box_mask > 0], reduction="mean")
                loss_cls = F.binary_cross_entropy_with_logits(
                    pred_cls[box_mask > 0], cls_target[box_mask > 0], reduction="mean"
                )
            else:
                loss_box = torch.tensor(0.0, device=pred.device)
                loss_cls = torch.tensor(0.0, device=pred.device)

            total_loss = total_loss + self.lambda_box * loss_box + self.lambda_obj * loss_obj + self.lambda_cls * loss_cls
        return total_loss


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        anchors: Dict[str, torch.Tensor],
        figo_class_weights: Optional[torch.Tensor] = None,
        image_size: int = 512,
    ) -> None:
        super().__init__()
        self.det_loss = YOLOv3Loss(anchors=anchors, image_size=image_size, num_classes=1)
        self.bce_seg = nn.BCEWithLogitsLoss()
        self.shape_ce = nn.CrossEntropyLoss()
        self.figo_ce = nn.CrossEntropyLoss(weight=figo_class_weights)
        self.size_mse = nn.MSELoss()
        self.ecc_mse = nn.MSELoss()

    def forward(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_det = self.det_loss(outputs["det_outputs"], targets)
        l_seg = dice_loss(outputs["seg_logits"], targets["mask"]) + self.bce_seg(outputs["seg_logits"], targets["mask"])
        l_shape = self.shape_ce(outputs["shape_logits"], targets["shape_label"])
        l_figo = self.figo_ce(outputs["figo_logits"], targets["figo_label"])
        l_size = self.size_mse(outputs["size_pred"].squeeze(1), targets["size_cm"])
        l_ecc = self.ecc_mse(outputs["ecc_pred"].squeeze(1), targets["eccentricity"])
        l_morph = l_shape + l_figo + l_size + l_ecc

        total = 1.0 * l_det + 2.0 * l_seg + 0.5 * l_morph
        loss_dict = {
            "total": float(total.detach().item()),
            "det": float(l_det.detach().item()),
            "seg": float(l_seg.detach().item()),
            "morph": float(l_morph.detach().item()),
        }
        return total, loss_dict
