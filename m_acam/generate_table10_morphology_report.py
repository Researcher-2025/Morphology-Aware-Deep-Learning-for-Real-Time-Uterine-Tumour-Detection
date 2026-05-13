"""
Table 10: Morphology-stratified segmentation Dice (M-ACAM vs baseline).

Rows: Round (e < τ), Elongated (e ≥ τ), Lobulated — with counts, mean Dice per model,
relative improvement (%), and optional paired t-test on the highlighted subgroup (default: elongated).

Modes (config JSON):
  - static: emit manuscript-style table from numeric rows in config (no GPU/data).
  - measure: run Table-8-style Dice inference on test split for two MACAM checkpoints + metadata CSV.

Config: m_acam/paper_tables/table10_morphology.example.json

Usage:
  python -m m_acam.generate_table10_morphology_report --config m_acam/paper_tables/table10_morphology.example.json

Measure mode: set "mode": "measure" in the JSON and include metadata_csv, checkpoint_macam,
checkpoint_baseline (same metadata rules as Table 8: round / elongated / lobulated via CSV fields
or eccentricity with threshold 0.7 in load_metadata). Row labels use eccentricity_threshold for display only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from m_acam.dataset import UterineFibroidDataset, collate_fn
from m_acam.generate_table8_segmentation_report import (
    bootstrap_ci,
    infer_dice_map,
    load_metadata,
)
from m_acam.model import MACAM
from m_acam.utils import set_global_seed

try:
    from scipy.stats import ttest_rel  # type: ignore
except Exception:  # pragma: no cover
    ttest_rel = None


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def morphology_subgroup(
    key: str,
    label: str,
    ids: List[str],
    dice_macam: Dict[str, float],
    dice_base: Dict[str, float],
    run_ttest: bool,
) -> Dict[str, Any]:
    va = [dice_macam[i] for i in ids if i in dice_macam and i in dice_base]
    vb = [dice_base[i] for i in ids if i in dice_macam and i in dice_base]
    n = len(va)
    if n == 0:
        return {
            "key": key,
            "label": label,
            "count": 0,
            "macam_dice": float("nan"),
            "baseline_dice": float("nan"),
            "improvement_pct": float("nan"),
            "ci95_macam": [float("nan"), float("nan")],
            "p_paired_ttest": float("nan"),
        }

    ma, mb = float(np.mean(va)), float(np.mean(vb))
    imp = ((ma - mb) / max(mb, 1e-8)) * 100.0
    ci = bootstrap_ci(va)
    p_t = float("nan")
    if run_ttest and ttest_rel is not None and n > 1:
        try:
            p_t = float(ttest_rel(va, vb).pvalue)
        except Exception:
            p_t = float("nan")

    return {
        "key": key,
        "label": label,
        "count": n,
        "macam_dice": ma,
        "baseline_dice": mb,
        "improvement_pct": imp,
        "ci95_macam": [ci[0], ci[1]],
        "p_paired_ttest": p_t,
    }


def build_rows_measure(cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau = float(cfg.get("eccentricity_threshold", 0.7))
    meta_path = cfg.get("metadata_csv") or args.metadata_csv
    ck_m = cfg.get("checkpoint_macam") or args.checkpoint_macam
    ck_b = cfg.get("checkpoint_baseline") or args.checkpoint_baseline
    if not meta_path or not ck_m or not ck_b:
        raise SystemExit("measure mode requires metadata_csv, checkpoint_macam, checkpoint_baseline (config or CLI).")

    ds_root = cfg.get("dataset_root", args.dataset_root)
    val_ratio = float(cfg.get("val_ratio", args.val_ratio))
    test_ratio = float(cfg.get("test_ratio", args.test_ratio))
    batch_size = int(cfg.get("batch_size", args.batch_size))

    ds = UterineFibroidDataset.from_voc_folder(
        dataset_root=ds_root,
        split="test",
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=42,
        train=False,
        apply_augmentation=False,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    m1 = MACAM(pretrained=False, num_det_classes=1).to(device)
    m2 = MACAM(pretrained=False, num_det_classes=1).to(device)
    w1 = torch.load(ck_m, map_location=device)
    w2 = torch.load(ck_b, map_location=device)
    m1.load_state_dict(w1["model"], strict=False)
    m2.load_state_dict(w2["model"], strict=False)

    dice_macam = infer_dice_map(m1, loader, device)
    dice_base = infer_dice_map(m2, loader, device)
    meta = load_metadata(meta_path)

    ids_all = [i for i in dice_macam if i in dice_base]

    ids_round = [i for i in ids_all if meta.get(i, {}).get("morphology") == "round"]
    ids_elong = [i for i in ids_all if meta.get(i, {}).get("morphology") == "elongated"]
    ids_lob = [i for i in ids_all if meta.get(i, {}).get("morphology") == "lobulated"]

    highlight = str(cfg.get("highlight_morphology", "elongated"))
    rows_raw = [
        morphology_subgroup(
            "round",
            f"Round (e < {tau})",
            ids_round,
            dice_macam,
            dice_base,
            run_ttest=False,
        ),
        morphology_subgroup(
            "elongated",
            f"Elongated (e ≥ {tau})",
            ids_elong,
            dice_macam,
            dice_base,
            run_ttest=(highlight == "elongated"),
        ),
        morphology_subgroup(
            "lobulated",
            "Lobulated",
            ids_lob,
            dice_macam,
            dice_base,
            run_ttest=(highlight == "lobulated"),
        ),
    ]

    rows: List[Dict[str, Any]] = []
    for r in rows_raw:
        sig = bool(
            r["key"] == highlight
            and np.isfinite(r.get("p_paired_ttest", float("nan")))
            and r["p_paired_ttest"] < float(cfg.get("significance_level", 0.001))
        )
        rows.append(
            {
                "key": r["key"],
                "label": r["label"],
                "count": r["count"],
                "macam_dice": r["macam_dice"],
                "baseline_dice": r["baseline_dice"],
                "improvement_pct": r["improvement_pct"],
                "ci95_macam": r["ci95_macam"],
                "p_paired_ttest": r["p_paired_ttest"],
                "significant": sig,
            }
        )

    notes = {
        "mode": "measure",
        "test_images_with_both_models": len(ids_all),
        "eccentricity_threshold": tau,
        "paired_test": "scipy.stats.ttest_rel on per-image Dice (highlight row only)",
        "ttest_available": ttest_rel is not None,
    }
    return rows, notes


def build_rows_static(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows_in = cfg.get("rows") or []
    rows: List[Dict[str, Any]] = []
    for r in rows_in:
        rows.append(
            {
                "key": str(r.get("key", "")),
                "label": str(r.get("label", r.get("key", ""))),
                "count": int(r.get("count", 0)),
                "macam_dice": float(r["macam_dice"]),
                "baseline_dice": float(r["baseline_dice"]),
                "improvement_pct": float(r.get("improvement_pct", 0.0)),
                "significant": bool(r.get("significant", False)),
                "p_paired_ttest": float(r["p_paired_ttest"]) if r.get("p_paired_ttest") is not None else None,
            }
        )
    notes = {"mode": "static", "paired_test": "values taken from config (no inference)"}
    return rows, notes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Table 10 morphology-stratified Dice report")
    p.add_argument("--config", type=str, default="m_acam/paper_tables/table10_morphology.example.json")
    p.add_argument("--dataset-root", type=str, default="Dataset")
    p.add_argument("--metadata-csv", type=str, default="")
    p.add_argument("--checkpoint-macam", type=str, default="")
    p.add_argument("--checkpoint-baseline", type=str, default="")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="checkpoints/table10_report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(42)
    cv2.setNumThreads(0)

    cfg = load_config(args.config)
    mode = str(cfg.get("mode", "static")).lower()
    baseline_name = str(cfg.get("baseline_name", "Swin-Unet"))
    highlight = str(cfg.get("highlight_morphology", "elongated"))
    sig_note = str(cfg.get("significance_note", "p < 0.001, validates morphology-aware attention hypothesis"))

    if mode == "measure":
        rows, notes = build_rows_measure(cfg, args)
    else:
        rows, notes = build_rows_static(cfg)

    payload: Dict[str, Any] = {
        "table": "Table 10: Morphology-Stratified Performance",
        "baseline_name": baseline_name,
        "highlight_morphology": highlight,
        "footnotes": {
            "significance": sig_note,
            "highlight_paired_ttest": next(
                (r.get("p_paired_ttest") for r in rows if r.get("key") == highlight),
                None,
            ),
        },
        "rows": rows,
        "notes": notes,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def json_default(o: Any) -> Any:
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        raise TypeError

    (out_dir / "table10_morphology.json").write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )

    with open(out_dir / "table10_morphology.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "morphology",
                "count",
                "macam_dice",
                "baseline_dice",
                "improvement_pct",
                "significant",
                "p_paired_ttest",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["label"],
                    r["count"],
                    r["macam_dice"],
                    r["baseline_dice"],
                    r["improvement_pct"],
                    r.get("significant", False),
                    r.get("p_paired_ttest", ""),
                ]
            )

    md: List[str] = [
        "## Table 10: Morphology-Stratified Performance (Critical Finding)",
        "",
        f"| Morphology | Count | M-ACAM Dice | {baseline_name} Dice | Improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        imp = float(r["improvement_pct"])
        imp_body = f"{imp:+.1f}%"
        star = "*" if r.get("significant") else ""
        imp_disp = f"{imp_body}{star}"
        line = (
            f"| {r['label']} | {r['count']} | {r['macam_dice']:.3f} | "
            f"{r['baseline_dice']:.3f} | {imp_disp} |"
        )
        if r.get("key") == highlight:
            line = line.replace(
                f"| {r['label']} |",
                f"| **{r['label']}** |",
                1,
            )
            line = line.replace(f"| {r['macam_dice']:.3f} |", f"| **{r['macam_dice']:.3f}** |", 1)
            line = line.replace(f"| {imp_disp} |", f"| **{imp_body}**{star} |", 1)
        md.append(line)

    md.append("")
    md.append(f"_\\* {sig_note}_")
    p_h = payload["footnotes"].get("highlight_paired_ttest")
    if p_h is not None and np.isfinite(float(p_h)):
        md.append(f"_Paired t-test (highlight subgroup): p = {float(p_h):.4g}_")

    (out_dir / "table10_morphology.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(payload, indent=2, default=json_default))
    print(f"Wrote {out_dir / 'table10_morphology.json'}")


if __name__ == "__main__":
    main()
