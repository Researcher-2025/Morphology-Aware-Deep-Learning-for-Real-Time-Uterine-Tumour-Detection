import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from m_acam.dataset import UterineFibroidDataset, collate_fn
from m_acam.metrics.detection_metrics import decode_detections
from m_acam.metrics.voc_ap import box_iou_np, compute_ap_voc, operating_point_f1
from m_acam.model import MACAM
from m_acam.utils import set_global_seed, to_device


def mcnemar_exact_p(correct_a: np.ndarray, correct_b: np.ndarray) -> Tuple[int, int, float]:
    b = int(np.sum((correct_a == 1) & (correct_b == 0)))
    c = int(np.sum((correct_a == 0) & (correct_b == 1)))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return b, c, float(min(p, 1.0))


def auc_bootstrap_diff_ci(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = y_true[idx]
        if y.min() == y.max():
            continue
        da = roc_auc_score(y, score_a[idx])
        db = roc_auc_score(y, score_b[idx])
        diffs.append(da - db)
    if not diffs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(np.array(diffs), [2.5, 97.5])
    return float(lo), float(hi)


def infer_detection_outputs(
    model: MACAM,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    min_confidence: float,
    nms_iou: float,
) -> Dict:
    gt_boxes: List[Optional[np.ndarray]] = []
    image_ids: List[str] = []
    y_true: List[int] = []
    pred_boxes: List[np.ndarray] = []
    pred_scores: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer", leave=False):
            b = to_device(batch, device)
            out = model(b["image"])
            dec = decode_detections(
                out["det_outputs"],
                model.anchors,
                conf_thresh=min_confidence,
                nms_iou=nms_iou,
                image_size=image_size,
            )
            for i, (boxes_t, scores_t) in enumerate(dec):
                has_obj = int(b["labels"][i][0].item() > 0)
                y_true.append(has_obj)
                image_ids.append(b["image_id"][i])
                if has_obj:
                    gt_boxes.append(b["boxes"][i][0].detach().cpu().numpy().astype(np.float64))
                else:
                    gt_boxes.append(None)

                if boxes_t.numel() == 0:
                    pred_boxes.append(np.zeros((0, 4), dtype=np.float64))
                    pred_scores.append(np.zeros((0,), dtype=np.float64))
                else:
                    pred_boxes.append(boxes_t.detach().cpu().numpy().astype(np.float64))
                    pred_scores.append(scores_t.detach().cpu().numpy().astype(np.float64))

    flat = []
    for img_i, (boxes, scores) in enumerate(zip(pred_boxes, pred_scores)):
        for j in range(len(scores)):
            flat.append((img_i, float(scores[j]), boxes[j]))

    map50, _, _ = compute_ap_voc(flat, gt_boxes, iou_threshold=0.5)
    map75, _, _ = compute_ap_voc(flat, gt_boxes, iou_threshold=0.75)
    return {
        "image_ids": image_ids,
        "y_true": np.array(y_true, dtype=np.int32),
        "gt_boxes": gt_boxes,
        "pred_boxes": pred_boxes,
        "pred_scores": pred_scores,
        "map50": float(map50),
        "map75": float(map75),
    }


def per_image_summary(
    gt_boxes: List[Optional[np.ndarray]],
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    conf_operating: float,
    match_iou: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_scores = np.zeros(len(gt_boxes), dtype=np.float64)
    correct = np.zeros(len(gt_boxes), dtype=np.int32)
    best_iou = np.zeros(len(gt_boxes), dtype=np.float64)
    for i, (gt, boxes, scores) in enumerate(zip(gt_boxes, pred_boxes, pred_scores)):
        if len(scores):
            j = int(np.argmax(scores))
            max_scores[i] = float(scores[j])
            if gt is not None and max_scores[i] >= conf_operating:
                iou = box_iou_np(boxes[j], gt)
                best_iou[i] = iou
                correct[i] = int(iou >= match_iou)
        else:
            max_scores[i] = 0.0
            correct[i] = int(gt is None)
    return max_scores, correct, best_iou


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Paired detection model comparison for manuscript reporting")
    p.add_argument("--dataset-root", type=str, default="Dataset")
    p.add_argument("--checkpoint-a", type=str, required=True, help="Model A checkpoint path")
    p.add_argument("--checkpoint-b", type=str, required=True, help="Model B checkpoint path")
    p.add_argument("--name-a", type=str, default="M-ACAM")
    p.add_argument("--name-b", type=str, default="YOLOv3")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--min-confidence", type=float, default=1e-5)
    p.add_argument("--nms-iou-threshold", type=float, default=0.45)
    p.add_argument("--conf-operating", type=float, default=0.25)
    p.add_argument("--match-iou-operating", type=float, default=0.5)
    p.add_argument("--out-dir", type=str, default="checkpoints/paired_detection_compare")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(42)
    cv2.setNumThreads(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = UterineFibroidDataset.from_voc_folder(
        dataset_root=args.dataset_root,
        split="test",
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=42,
        train=False,
        apply_augmentation=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model_a = MACAM(pretrained=False, num_det_classes=1).to(device)
    model_b = MACAM(pretrained=False, num_det_classes=1).to(device)
    ckpt_a = torch.load(args.checkpoint_a, map_location=device)
    ckpt_b = torch.load(args.checkpoint_b, map_location=device)
    model_a.load_state_dict(ckpt_a["model"], strict=False)
    model_b.load_state_dict(ckpt_b["model"], strict=False)

    out_a = infer_detection_outputs(
        model_a, loader, device, args.image_size, args.min_confidence, args.nms_iou_threshold
    )
    out_b = infer_detection_outputs(
        model_b, loader, device, args.image_size, args.min_confidence, args.nms_iou_threshold
    )

    y_true = out_a["y_true"]
    if not np.array_equal(y_true, out_b["y_true"]):
        raise RuntimeError("Ground truth mismatch between model runs.")

    p_a, r_a, f1_a, tp_a, fp_a, fn_a = operating_point_f1(
        out_a["gt_boxes"],
        out_a["pred_boxes"],
        out_a["pred_scores"],
        score_thresh=args.conf_operating,
        match_iou=args.match_iou_operating,
    )
    p_b, r_b, f1_b, tp_b, fp_b, fn_b = operating_point_f1(
        out_b["gt_boxes"],
        out_b["pred_boxes"],
        out_b["pred_scores"],
        score_thresh=args.conf_operating,
        match_iou=args.match_iou_operating,
    )

    score_a, corr_a, iou_a = per_image_summary(
        out_a["gt_boxes"], out_a["pred_boxes"], out_a["pred_scores"], args.conf_operating, args.match_iou_operating
    )
    score_b, corr_b, iou_b = per_image_summary(
        out_b["gt_boxes"], out_b["pred_boxes"], out_b["pred_scores"], args.conf_operating, args.match_iou_operating
    )

    auc_a = float(roc_auc_score(y_true, score_a)) if y_true.min() != y_true.max() else float("nan")
    auc_b = float(roc_auc_score(y_true, score_b)) if y_true.min() != y_true.max() else float("nan")
    auc_diff = auc_a - auc_b if not np.isnan(auc_a) and not np.isnan(auc_b) else float("nan")
    auc_ci_lo, auc_ci_hi = auc_bootstrap_diff_ci(y_true, score_a, score_b, n_boot=2000, seed=42)

    b, c, p_mcnemar = mcnemar_exact_p(corr_a, corr_b)

    payload = {
        "names": {"a": args.name_a, "b": args.name_b},
        "metrics_a": {
            "map50": out_a["map50"],
            "map75": out_a["map75"],
            "precision_op": p_a,
            "recall_op": r_a,
            "f1_op": f1_a,
            "auc_image_level": auc_a,
            "tp": tp_a,
            "fp": fp_a,
            "fn": fn_a,
        },
        "metrics_b": {
            "map50": out_b["map50"],
            "map75": out_b["map75"],
            "precision_op": p_b,
            "recall_op": r_b,
            "f1_op": f1_b,
            "auc_image_level": auc_b,
            "tp": tp_b,
            "fp": fp_b,
            "fn": fn_b,
        },
        "delta_a_minus_b": {
            "map50": out_a["map50"] - out_b["map50"],
            "map75": out_a["map75"] - out_b["map75"],
            "precision_op": p_a - p_b,
            "recall_op": r_a - r_b,
            "f1_op": f1_a - f1_b,
            "auc_image_level": auc_diff,
        },
        "significance": {
            "mcnemar": {
                "b_a_correct_b_wrong": b,
                "c_a_wrong_b_correct": c,
                "p_value_exact": p_mcnemar,
            },
            "auc_diff_bootstrap_95ci": [auc_ci_lo, auc_ci_hi],
        },
        "settings": {
            "conf_operating": args.conf_operating,
            "match_iou_operating": args.match_iou_operating,
            "nms_iou_threshold": args.nms_iou_threshold,
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paired_detection_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with open(out_dir / "paired_detection_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "image_id",
                "y_true",
                "score_a",
                "score_b",
                "correct_a",
                "correct_b",
                "best_iou_a",
                "best_iou_b",
            ]
        )
        for i in range(len(y_true)):
            w.writerow(
                [
                    out_a["image_ids"][i],
                    int(y_true[i]),
                    float(score_a[i]),
                    float(score_b[i]),
                    int(corr_a[i]),
                    int(corr_b[i]),
                    float(iou_a[i]),
                    float(iou_b[i]),
                ]
            )

    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_dir / 'paired_detection_summary.json'}")
    print(f"Wrote {out_dir / 'paired_detection_per_image.csv'}")


if __name__ == "__main__":
    main()

