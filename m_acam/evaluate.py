import argparse

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from m_acam.dataset import UterineFibroidDataset, collate_fn
from m_acam.metrics import average_precision_11_point, box_iou_xyxy, decode_detections, write_metrics_csv, write_metrics_json
from m_acam.model import MACAM
from m_acam.utils import set_global_seed, to_device

def evaluate_detection(args: argparse.Namespace) -> None:
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
    model.eval()

    tp = 0
    fp = 0
    fn = 0
    iou_matches = []
    ap_rows = []
    total_gt = 0
    image_tp = 0
    image_tn = 0
    image_fp = 0
    image_fn = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = to_device(batch, device)
            outputs = model(batch["image"])
            preds = decode_detections(
                outputs["det_outputs"],
                model.anchors,
                conf_thresh=args.conf_threshold,
                nms_iou=args.nms_iou_threshold,
            )
            for i, (pred_boxes, pred_scores) in enumerate(preds):
                gt_box = batch["boxes"][i][0]
                has_gt = batch["labels"][i][0].item() > 0
                pred_positive = pred_boxes.shape[0] > 0
                if has_gt:
                    total_gt += 1

                if has_gt and pred_positive:
                    image_tp += 1
                elif has_gt and not pred_positive:
                    image_fn += 1
                elif (not has_gt) and pred_positive:
                    image_fp += 1
                else:
                    image_tn += 1

                if not has_gt:
                    ap_rows.extend([(float(s.item()), 0) for s in pred_scores])
                    fp += int(pred_boxes.shape[0])
                    continue

                if pred_boxes.shape[0] == 0:
                    fn += 1
                    continue

                ious = torch.tensor([box_iou_xyxy(pb, gt_box) for pb in pred_boxes], device=device)
                best_iou, best_idx = torch.max(ious, dim=0)
                matched_idx = int(best_idx.item())
                if float(best_iou.item()) >= args.match_iou_threshold:
                    tp += 1
                    iou_matches.append(float(best_iou.item()))
                    fp += int(pred_boxes.shape[0] - 1)
                    for j, s in enumerate(pred_scores):
                        ap_rows.append((float(s.item()), 1 if j == matched_idx else 0))
                else:
                    fp += int(pred_boxes.shape[0])
                    fn += 1
                    ap_rows.extend([(float(s.item()), 0) for s in pred_scores])

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    mean_iou = sum(iou_matches) / max(len(iou_matches), 1)
    total_images = image_tp + image_tn + image_fp + image_fn
    accuracy = (image_tp + image_tn) / max(total_images, 1)
    specificity = image_tn / max(image_tn + image_fp, 1)
    sensitivity = image_tp / max(image_tp + image_fn, 1)
    map50 = average_precision_11_point(ap_rows, total_gt)

    print("Detection evaluation (test split)")
    print(f"TP={tp} FP={fp} FN={fn}")
    print(f"Image-level TP={image_tp} TN={image_tn} FP={image_fp} FN={image_fn}")
    print(f"Accuracy={accuracy:.4f}")
    print(f"Precision={precision:.4f}")
    print(f"Recall={recall:.4f}")
    print(f"F1={f1:.4f}")
    print(f"Sensitivity={sensitivity:.4f}")
    print(f"Specificity={specificity:.4f}")
    print(f"mAP@0.5={map50:.4f}")
    print(f"Mean IoU (matched)={mean_iou:.4f}")

    metrics = {
        "bbox_level": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": map50,
            "mean_iou_matched": mean_iou,
        },
        "image_level": {
            "tp": image_tp,
            "tn": image_tn,
            "fp": image_fp,
            "fn": image_fn,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
        },
        "thresholds": {
            "confidence": args.conf_threshold,
            "nms_iou": args.nms_iou_threshold,
            "match_iou": args.match_iou_threshold,
        },
    }

    if args.output_json:
        write_metrics_json(metrics, args.output_json)
        print(f"Saved metrics JSON: {args.output_json}")
    if args.output_csv:
        write_metrics_csv(metrics, args.output_csv)
        print(f"Saved metrics CSV: {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("M-ACAM Detection Evaluation")
    parser.add_argument("--dataset-root", type=str, default="Dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save metrics JSON report.")
    parser.add_argument("--output-csv", type=str, default="", help="Optional path to save metrics CSV report.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_detection(parse_args())
