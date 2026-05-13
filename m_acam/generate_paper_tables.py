import argparse
import json
from pathlib import Path


def generate_detection_table(metrics_json: str, out_md: str) -> None:
    with open(metrics_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    det = data.get("bbox_level", {})
    img = data.get("image_level", {})
    th = data.get("thresholds", {})

    lines = [
        "# Detection Metrics Table",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {img.get('accuracy', 0.0):.4f} |",
        f"| Precision | {det.get('precision', 0.0):.4f} |",
        f"| Recall | {det.get('recall', 0.0):.4f} |",
        f"| F1-score | {det.get('f1', 0.0):.4f} |",
        f"| Sensitivity | {img.get('sensitivity', 0.0):.4f} |",
        f"| Specificity | {img.get('specificity', 0.0):.4f} |",
        f"| mAP@0.5 | {det.get('map50', 0.0):.4f} |",
        f"| Mean IoU | {det.get('mean_iou_matched', 0.0):.4f} |",
        "",
        "## Evaluation Thresholds",
        "",
        f"- Confidence threshold: {th.get('confidence', 0.25)}",
        f"- NMS IoU threshold: {th.get('nms_iou', 0.45)}",
        f"- Match IoU threshold: {th.get('match_iou', 0.5)}",
    ]

    out_path = Path(out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote markdown table: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate paper-ready table from metrics JSON")
    parser.add_argument("--metrics-json", type=str, required=True)
    parser.add_argument("--out-md", type=str, default="checkpoints/paper_detection_table.md")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_detection_table(args.metrics_json, args.out_md)
