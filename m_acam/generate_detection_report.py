"""
Full detection report aligned with manuscript Table 7 / Fig. 7–style outputs:

- mAP@0.5, mAP@0.75 (VOC-style, one GT box per image)
- Precision, Recall, F1 at operating-point (score + IoU thresholds)
- Image-level ROC AUC (max detection score vs presence label)
- Stratified mAP@0.5 by fibroid diameter (requires CSV image_id,diameter_cm)
- Optional FPS (same benchmark style as performance table)
- Writes JSON, CSV, Markdown; optional ROC/PR plots

Example diameter CSV: see m_acam/paper_tables/diameter.example.csv

Usage:
  python -m m_acam.generate_detection_report --checkpoint checkpoints/best_model.pt --dataset-root Dataset
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import cv2

from m_acam.dataset import UterineFibroidDataset, collate_fn
from m_acam.metrics.detection_curves import image_level_roc, max_f1_from_pr
from m_acam.metrics.detection_metrics import decode_detections
from m_acam.metrics.voc_ap import compute_ap_voc, operating_point_f1
from m_acam.model import MACAM
from m_acam.paper_tables.runtime_bench import benchmark_fps, peak_inference_memory_mb_cuda
from m_acam.utils import set_global_seed, to_device


def load_diameter_csv(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = row.get("image_id", row.get("id", "")).strip()
            if not iid:
                continue
            val = row.get("diameter_cm", row.get("diameter", row.get("size_cm", "")))
            if val is None or str(val).strip() == "":
                continue
            out[iid] = float(val)
    return out


def size_bucket(cm: float) -> str:
    if cm < 3.0:
        return "small_lt_3cm"
    if cm <= 8.0:
        return "medium_3_8cm"
    return "large_gt_8cm"


def collect_predictions(
    model: MACAM,
    loader: DataLoader,
    device: torch.device,
    min_confidence: float,
    nms_iou: float,
    image_size: int,
) -> Tuple[
    List[Optional[np.ndarray]],
    List[Tuple[np.ndarray, np.ndarray]],
    List[str],
    List[int],
]:
    """Returns gt_boxes (None if negative), pred (boxes,scores) per image, image_ids, y_true presence."""
    gt_list: List[Optional[np.ndarray]] = []
    preds_per: List[Tuple[np.ndarray, np.ndarray]] = []
    image_ids: List[str] = []
    y_presence: List[int] = []

    model.eval()
    anchors = {k: v.to(device) for k, v in model.anchors.items()}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Detection inference"):
            b = to_device(batch, device)
            imgs = b["image"]
            outputs = model(imgs)
            dec = decode_detections(
                outputs["det_outputs"],
                anchors,
                conf_thresh=min_confidence,
                nms_iou=nms_iou,
                image_size=image_size,
            )
            bsz = imgs.shape[0]
            for i in range(bsz):
                gt = b["boxes"][i][0].detach().cpu().numpy()
                if b["labels"][i][0].item() <= 0:
                    gt_list.append(None)
                    y_presence.append(0)
                else:
                    gt_list.append(gt.astype(np.float64))
                    y_presence.append(1)

                image_ids.append(b["image_id"][i])
                boxes, scores = dec[i]
                preds_per.append(
                    (
                        boxes.detach().cpu().numpy().astype(np.float64) if boxes.numel() else np.zeros((0, 4)),
                        scores.detach().cpu().numpy().astype(np.float64) if scores.numel() else np.zeros((0,)),
                    )
                )

    return gt_list, preds_per, image_ids, y_presence


def flatten_detections(
    preds_per: List[Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[int, float, np.ndarray]]:
    dets: List[Tuple[int, float, np.ndarray]] = []
    for img_i, (boxes, scores) in enumerate(preds_per):
        for j in range(len(scores)):
            dets.append((img_i, float(scores[j]), boxes[j]))
    return dets


def max_scores(preds_per: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    out = np.zeros(len(preds_per), dtype=np.float64)
    for i, (_, scores) in enumerate(preds_per):
        if len(scores):
            out[i] = float(np.max(scores))
    return out


def stratified_indices(
    image_ids: Sequence[str],
    diameters: Dict[str, float],
    bucket: str,
) -> List[int]:
    idxs: List[int] = []
    for i, iid in enumerate(image_ids):
        d = diameters.get(iid)
        if d is None:
            continue
        if size_bucket(d) == bucket:
            idxs.append(i)
    return idxs


def subset_detection_task(
    indices: List[int],
    gt_list: Sequence[Optional[np.ndarray]],
    dets_flat: Sequence[Tuple[int, float, np.ndarray]],
) -> Tuple[List[Optional[np.ndarray]], List[Tuple[int, float, np.ndarray]]]:
    sidx = set(indices)
    gt_sub = [gt_list[i] for i in indices]
    old_to_new = {old: new for new, old in enumerate(indices)}
    dets_remapped = [
        (old_to_new[i], s, b)
        for i, s, b in dets_flat
        if i in sidx
    ]
    return gt_sub, dets_remapped


def try_plot_roc_pr(
    out_png: str,
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    precisions: np.ndarray,
    recalls: np.ndarray,
    title_prefix: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib for --plot.") from exc

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, "k-")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"{title_prefix} ROC (AUC={roc_auc:.3f})")

    axes[1].plot(recalls, precisions, "k-")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{title_prefix} Precision–Recall (from AP sort)")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Full detection report (Table 7 style)")
    p.add_argument("--dataset-root", type=str, default="Dataset")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=512)

    p.add_argument("--min-confidence", type=float, default=1e-5, help="Keep low for AP; internal filter before NMS.")
    p.add_argument("--nms-iou-threshold", type=float, default=0.45)
    p.add_argument("--conf-operating", type=float, default=0.25, help="Score threshold for P/R/F1 operating point.")
    p.add_argument("--match-iou-operating", type=float, default=0.5, help="IoU for TP at operating point.")

    p.add_argument("--diameter-csv", type=str, default="", help="Optional CSV: image_id,diameter_cm for strata.")

    p.add_argument("--out-dir", type=str, default="checkpoints/detection_report")
    p.add_argument("--plot", type=str, default="", help="Optional PNG path for ROC + PR panels.")
    p.add_argument("--benchmark-fps", action="store_true", help="Measure MACAM FPS on this device.")

    return p.parse_args()


def run(args: argparse.Namespace) -> None:
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

    model = MACAM(pretrained=False, num_det_classes=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    gt_list, preds_per, image_ids, y_presence = collect_predictions(
        model,
        loader,
        device,
        min_confidence=args.min_confidence,
        nms_iou=args.nms_iou_threshold,
        image_size=args.image_size,
    )

    dets_flat = flatten_detections(preds_per)

    map50, prec50, rec50 = compute_ap_voc(dets_flat, gt_list, iou_threshold=0.5)
    map75, _prec75, _rec75 = compute_ap_voc(dets_flat, gt_list, iou_threshold=0.75)

    p_op, r_op, f1_op, tp, fp, fn = operating_point_f1(
        gt_list,
        [p[0] for p in preds_per],
        [p[1] for p in preds_per],
        score_thresh=args.conf_operating,
        match_iou=args.match_iou_operating,
    )

    scores_max = max_scores(preds_per)
    y = np.array(y_presence, dtype=np.int32)
    if y.sum() in (0, len(y)):
        roc_auc = float("nan")
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
    else:
        roc_auc, fpr, tpr, _ = image_level_roc(y, scores_max)
    f1_max, _j = max_f1_from_pr(prec50, rec50) if prec50.size else (0.0, 0)

    diameters: Dict[str, float] = {}
    if args.diameter_csv:
        diameters = load_diameter_csv(args.diameter_csv)

    strata: Dict[str, float] = {}
    strata_notes: Dict[str, str] = {}
    for label, bucket_key in [
        ("small_lt_3cm", "small_lt_3cm"),
        ("medium_3_8cm", "medium_3_8cm"),
        ("large_gt_8cm", "large_gt_8cm"),
    ]:
        if not diameters:
            strata_notes[label] = "No diameter CSV; skipped."
            continue
        idxs = stratified_indices(image_ids, diameters, bucket_key)
        if not idxs:
            strata_notes[label] = "No images in stratum (check CSV IDs vs split)."
            continue
        gt_s, dets_s = subset_detection_task(idxs, gt_list, dets_flat)
        ap_s, _, _ = compute_ap_voc(dets_s, gt_s, iou_threshold=0.5)
        strata[label] = float(ap_s)

    fps_val = None
    mem_note = ""
    if args.benchmark_fps:
        inp = (1, 1, args.image_size, args.image_size)
        fps_val = benchmark_fps(model, device, inp, warmup=10, repeats=50)
        if device.type == "cuda":
            mem_note = f"Peak CUDA forward memory (MB): {peak_inference_memory_mb_cuda(model, device, inp):.2f}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "split": "test",
        "count_images": len(gt_list),
        "map50": map50,
        "map75": map75,
        "operating_point": {
            "confidence_threshold": args.conf_operating,
            "match_iou": args.match_iou_operating,
            "precision": p_op,
            "recall": r_op,
            "f1": f1_op,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "image_level_roc_auc": roc_auc,
        "ap_curve_f1_max_proxy": f1_max,
        "stratified_map50": strata,
        "stratified_notes": strata_notes,
        "inference": {"fps": fps_val, "note": mem_note},
        "diameter_csv_used": bool(diameters),
    }

    (out_dir / "table7_detection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with open(out_dir / "table7_detection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["count", len(gt_list)])
        w.writerow(["mAP@0.5", f"{map50:.6f}"])
        w.writerow(["mAP@0.75", f"{map75:.6f}"])
        w.writerow(["precision_op", f"{p_op:.6f}"])
        w.writerow(["recall_op", f"{r_op:.6f}"])
        w.writerow(["f1_op", f"{f1_op:.6f}"])
        w.writerow(["roc_auc_image_level", f"{roc_auc:.6f}"])
        w.writerow(["f1_max_from_AP_curve", f"{f1_max:.6f}"])
        if fps_val is not None:
            w.writerow(["fps", f"{fps_val:.2f}"])
        for k, v in strata.items():
            w.writerow([f"map50_{k}", f"{v:.6f}"])

    md_lines = [
        "## Detection performance (Table 7 style)",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Count | {len(gt_list)} |",
        f"| mAP@0.5 | {map50:.4f} |",
        f"| mAP@0.75 | {map75:.4f} |",
        f"| Precision (op) | {p_op:.4f} |",
        f"| Recall (op) | {r_op:.4f} |",
        f"| F1 (op) | {f1_op:.4f} |",
        f"| Image-level ROC AUC | {roc_auc:.4f} |",
        "",
    ]
    if strata:
        md_lines.extend(["### Stratified mAP@0.5 (diameter CSV)", ""])
        for k, v in strata.items():
            md_lines.append(f"- **{k}**: {v:.4f}")
        md_lines.append("")
    else:
        md_lines.append("_Stratified rows: provide `--diameter-csv` with `image_id,diameter_cm`._\n")

    (out_dir / "table7_detection.md").write_text("\n".join(md_lines), encoding="utf-8")

    if args.plot:
        try_plot_roc_pr(
            args.plot,
            fpr,
            tpr,
            roc_auc,
            prec50,
            rec50,
            "M-ACAM detection",
        )
        print(f"Wrote plot {args.plot}")

    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_dir / 'table7_detection.json'}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
