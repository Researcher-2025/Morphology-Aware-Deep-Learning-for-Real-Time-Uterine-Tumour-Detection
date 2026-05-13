"""
Generate a dynamic model-comparison report (e.g. manuscript Table 5 style).

Counts trainable (and total) parameters from instantiated PyTorch modules.
Baselines without a `builder` in comparison_registry.py are skipped with a note.

Usage (from project root):
  python -m m_acam.generate_model_comparison_report --out-dir checkpoints/paper_reports
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import torch.nn as nn

from m_acam.paper_tables.comparison_registry import DEFAULT_COMPARISON_REGISTRY, ModelComparisonSpec


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def params_to_millions(n: int) -> float:
    return round(n / 1_000_000.0, 2)


def row_from_spec(spec: ModelComparisonSpec, trainable_only: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "key": spec.key,
        "model": spec.display_name,
        "type": spec.task_type,
        "source": spec.source,
        "trainable_params": None,
        "total_params": None,
        "params_m_trainable": None,
        "params_m_total": None,
        "status": "skipped",
        "notes": "",
    }

    if spec.builder is None:
        out["notes"] = (
            "No builder registered — add a factory in "
            "`m_acam/paper_tables/comparison_registry.py` for this baseline."
        )
        return out

    try:
        model = spec.builder()
        model.eval()
        tp = count_parameters(model, trainable_only=True)
        tot = count_parameters(model, trainable_only=False)
        out["trainable_params"] = tp
        out["total_params"] = tot
        out["params_m_trainable"] = params_to_millions(tp)
        out["params_m_total"] = params_to_millions(tot)
        out["status"] = "ok"
        metric = params_to_millions(tp if trainable_only else tot)
        out["notes"] = f"Reported Params (M): {metric} ({'trainable' if trainable_only else 'total'})"
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["notes"] = f"Instantiation failed: {exc!s}"

    return out


def write_csv(rows: List[Dict[str, Any]], path: Path, trainable_primary: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "type",
        "params_m_primary",
        "trainable_params",
        "total_params",
        "source",
        "status",
        "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            primary = r.get("params_m_trainable") if trainable_primary else r.get("params_m_total")
            r_out = {
                "model": r["model"],
                "type": r["type"],
                "params_m_primary": primary if r["status"] == "ok" and primary is not None else "",
                "trainable_params": r.get("trainable_params", "") if r["status"] == "ok" else "",
                "total_params": r.get("total_params", "") if r["status"] == "ok" else "",
                "source": r["source"],
                "status": r["status"],
                "notes": r["notes"],
            }
            w.writerow(r_out)


def write_md(rows: List[Dict[str, Any]], path: Path, title: str, trainable_only: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kind = "trainable (M)" if trainable_only else "total (M)"
    lines = [
        f"## {title}",
        "",
        f"_Parameter column: **{kind}**. Match this definition with the manuscript Methods._",
        "",
        "| Model | Type | Params (M) | Source | Status | Notes |",
        "|---|---|---:|---|---|---|",
    ]
    for r in rows:
        pm = r.get("params_m_trainable") if trainable_only else r.get("params_m_total")
        pm_s = "" if r["status"] != "ok" or pm is None else str(pm)
        lines.append(
            f"| {r['model']} | {r['type']} | {pm_s} | {r['source']} | {r['status']} | {r['notes']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for spec in DEFAULT_COMPARISON_REGISTRY:
        r = row_from_spec(spec, trainable_only=args.trainable_only)
        rows.append(r)

    payload = {
        "title": args.title,
        "primary_metric": "trainable_params_m" if args.trainable_only else "total_params_m",
        "rows": rows,
    }

    json_path = out_dir / "model_comparison_report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out_dir / "model_comparison_report.csv"
    write_csv(rows, csv_path, trainable_primary=args.trainable_only)

    md_path = out_dir / "model_comparison_report.md"
    write_md(rows, md_path, title=args.title, trainable_only=args.trainable_only)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Dynamic model comparison report (parameter counts)")
    p.add_argument("--out-dir", type=str, default="checkpoints/paper_reports")
    p.add_argument(
        "--title",
        type=str,
        default="Table: M-ACAM vs baselines (parameter counts from implementation)",
    )
    p.add_argument(
        "--use-total-params",
        action="store_true",
        default=False,
        help="Use total parameter count as primary Params (M). Default is trainable-only.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.trainable_only = not args.use_total_params
    run(args)


if __name__ == "__main__":
    main()
