import argparse
import csv
from pathlib import Path

from m_acam.dataset import load_voc_samples


def build_template(dataset_root: str, output_csv: str) -> None:
    samples = load_voc_samples(dataset_root)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "shape", "size_cm", "figo", "eccentricity"])
        for s in samples:
            writer.writerow([s.get("image_id", Path(s["image"]).stem), "", "", "", ""])
    print(f"Wrote template: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Create morphology annotation template")
    parser.add_argument("--dataset-root", type=str, default="Dataset")
    parser.add_argument("--output-csv", type=str, default="m_acam/morphology_labels_template.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_template(args.dataset_root, args.output_csv)
