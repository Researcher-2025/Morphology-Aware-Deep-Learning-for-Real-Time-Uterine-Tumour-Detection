import csv
import json
from pathlib import Path
from typing import Dict


def write_metrics_json(metrics: Dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def write_metrics_csv(metrics: Dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for section, values in metrics.items():
        if not isinstance(values, dict):
            continue
        for k, v in values.items():
            rows.append({"section": section, "metric": k, "value": v})
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
