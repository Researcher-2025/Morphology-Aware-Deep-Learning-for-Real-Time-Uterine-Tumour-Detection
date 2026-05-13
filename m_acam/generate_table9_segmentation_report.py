"""
Table 9: Segmentation performance comparison (Dice, IoU, Precision, Recall, Boundary F1, HD95).

- Each model row is either `static` (paper numbers) or `checkpoint` (measured on test split via MACAM).
- Optional paired Wilcoxon + paired t-test on per-image Dice when both primary and reference have checkpoints.

Config: m_acam/paper_tables/table9_segmentation.example.json

Usage:
  python -m m_acam.generate_table9_segmentation_report --config m_acam/paper_tables/table9_segmentation.example.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from m_acam.dataset import UterineFibroidDataset, collate_fn
from m_acam.metrics.segmentation_extended import (
    boundary_f1,
    dice_iou,
    hd95_pixels,
    hd95_to_mm,
    pixel_precision_recall,
)
from m_acam.model import MACAM
from m_acam.utils import set_global_seed, to_device

try:
    from scipy.stats import ttest_rel, wilcoxon
except Exception:  # pragma: no cover
    ttest_rel = None
    wilcoxon = None


def bootstrap_mean_ci(values: List[float], n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = [float(arr[rng.integers(0, len(arr), len(arr))].mean()) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def evaluate_checkpoint(
    ckpt_path: str,
    loader: DataLoader,
    device: torch.device,
    spacing_mm: float,
    boundary_tol: int,
) -> Dict[str, Any]:
    model = MACAM(pretrained=False, num_det_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    dices: List[float] = []
    ious: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    bf1s: List[float] = []
    hd_mms: List[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Seg eval", leave=False):
            b = to_device(batch, device)
            logits = model(b["image"])["seg_logits"]
            prob = torch.sigmoid(logits)
            for i in range(prob.shape[0]):
                p = prob[i : i + 1]
                t = b["mask"][i : i + 1]
                d, j = dice_iou(p, t)
                pr, rc = pixel_precision_recall(p, t)
                bf = boundary_f1(p, t, tolerance_px=boundary_tol)
                hd_px = hd95_pixels(p, t)
                hd_mm = hd95_to_mm(hd_px, spacing_mm) if np.isfinite(hd_px) else float("nan")

                dices.append(d)
                ious.append(j)
                precs.append(pr)
                recs.append(rc)
                bf1s.append(bf)
                if np.isfinite(hd_mm):
                    hd_mms.append(hd_mm)

    def mean(xs: List[float]) -> float:
        return float(np.nanmean(xs)) if xs else float("nan")

    return {
        "dice": mean(dices),
        "iou": mean(ious),
        "precision": mean(precs),
        "recall": mean(recs),
        "boundary_f1": mean(bf1s),
        "hd95_mm": mean(hd_mms),
        "per_image_dice": dices,
        "dice_ci95": bootstrap_mean_ci(dices),
    }


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Table 9 segmentation comparison report")
    p.add_argument("--config", type=str, default="m_acam/paper_tables/table9_segmentation.example.json")
    p.add_argument("--dataset-root", type=str, default="Dataset")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="checkpoints/table9_report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(42)
    cv2.setNumThreads(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    spacing = float(cfg.get("pixel_spacing_mm", 0.1))
    boundary_tol = int(cfg.get("boundary_tolerance_px", 2))
    paired = cfg.get("paired_stats") or {}

    needs_eval = any(str(e.get("checkpoint", "") or "").strip() for e in cfg["models"])
    loader: Optional[DataLoader] = None
    if needs_eval:
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

    rows_out: Dict[str, Dict[str, Any]] = {}
    per_image_store: Dict[str, List[float]] = {}

    for entry in cfg["models"]:
        name = entry["name"]
        static = entry.get("static")
        ckpt = entry.get("checkpoint", "") or ""

        if ckpt:
            assert loader is not None
            metrics = evaluate_checkpoint(ckpt, loader, device, spacing, boundary_tol)
            rows_out[name] = {
                "source": "checkpoint",
                "checkpoint": ckpt,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "boundary_f1": metrics["boundary_f1"],
                "hd95_mm": metrics["hd95_mm"],
                "dice_ci95": metrics["dice_ci95"],
            }
            per_image_store[name] = metrics["per_image_dice"]
            continue

        if static:
            rows_out[name] = {
                "source": "static",
                **{k: float(static[k]) for k in ["dice", "iou", "precision", "recall", "boundary_f1", "hd95_mm"]},
            }
        else:
            rows_out[name] = {"source": "missing", "error": "Provide static or checkpoint"}

    # Order rows as in config
    ordered_names = [e["name"] for e in cfg["models"]]
    table_rows: List[Dict[str, Any]] = []
    for n in ordered_names:
        if n not in rows_out:
            continue
        row = dict(rows_out[n])
        row["name"] = n
        table_rows.append(row)

    # Best baseline (max dice) excluding primary
    primary_name = paired.get("primary_name", "M-ACAM (Ours)")
    cand = [(n, r["dice"]) for n, r in rows_out.items() if n != primary_name and "dice" in r and np.isfinite(r["dice"])]
    best_name, best_dice = max(cand, key=lambda x: x[1]) if cand else (None, float("nan"))

    primary = rows_out.get(primary_name, {})
    vs_row: Dict[str, Any] = {"name": "vs. Best Baseline", "source": "derived"}
    if best_name and np.isfinite(primary.get("dice", float("nan"))) and np.isfinite(best_dice):
        pd, bd = primary["dice"], best_dice
        vs_row["dice_pct"] = ((pd - bd) / max(bd, 1e-8)) * 100.0
        br = rows_out[best_name]
        if "iou" in primary and "iou" in br:
            vs_row["iou_pct"] = ((primary["iou"] - br["iou"]) / max(br["iou"], 1e-8)) * 100.0
        if "precision" in primary and "precision" in br:
            vs_row["precision_pct"] = ((primary["precision"] - br["precision"]) / max(br["precision"], 1e-8)) * 100.0
        if "recall" in primary and "recall" in br:
            vs_row["recall_pct"] = ((primary["recall"] - br["recall"]) / max(br["recall"], 1e-8)) * 100.0
        if "boundary_f1" in primary and "boundary_f1" in br:
            vs_row["boundary_f1_pct"] = ((primary["boundary_f1"] - br["boundary_f1"]) / max(br["boundary_f1"], 1e-8)) * 100.0
        if "hd95_mm" in primary and np.isfinite(primary.get("hd95_mm", float("nan"))) and np.isfinite(br.get("hd95_mm", float("nan"))):
            hb = br["hd95_mm"]
            vs_row["hd95_pct"] = ((primary["hd95_mm"] - hb) / max(hb, 1e-8)) * 100.0

    # Paired stats
    stats_block: Dict[str, Any] = {}
    ref_name = paired.get("reference_name")
    if (
        ref_name
        and primary_name in per_image_store
        and ref_name in per_image_store
        and wilcoxon is not None
        and ttest_rel is not None
    ):
        a = np.array(per_image_store[primary_name])
        b = np.array(per_image_store[ref_name])
        if len(a) == len(b) and len(a) > 1:
            try:
                w = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                stats_block["wilcoxon_dice_p"] = float(w.pvalue)
            except Exception:
                stats_block["wilcoxon_dice_p"] = float("nan")
            try:
                t = ttest_rel(a, b)
                stats_block["paired_t_dice"] = {"statistic": float(t.statistic), "pvalue": float(t.pvalue)}
                pooled = np.std(a - b, ddof=1)
                stats_block["cohens_d"] = float((np.mean(a - b)) / max(pooled, 1e-8))
            except Exception:
                pass

    dice_ci_primary = primary.get("dice_ci95") if primary.get("source") == "checkpoint" else None

    if ref_name and ref_name not in per_image_store:
        stats_block["paired_stats_note"] = (
            f"Wilcoxon/t-test skipped: '{ref_name}' has no checkpoint (add checkpoint to config for paired tests)."
        )

    payload = {
        "best_baseline_name": best_name,
        "table_rows": table_rows,
        "vs_best_baseline": vs_row,
        "footnotes": {
            "dice_ci95_macam": dice_ci_primary,
            "paired_reference": ref_name,
            **stats_block,
        },
        "pixel_spacing_mm": spacing,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table9_segmentation.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    with open(out_dir / "table9_segmentation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "dice", "iou", "precision", "recall", "boundary_f1", "hd95_mm", "source"])
        for r in table_rows:
            w.writerow(
                [
                    r["name"],
                    r.get("dice", ""),
                    r.get("iou", ""),
                    r.get("precision", ""),
                    r.get("recall", ""),
                    r.get("boundary_f1", ""),
                    r.get("hd95_mm", ""),
                    r.get("source", ""),
                ]
            )
        if vs_row.get("dice_pct") is not None:
            w.writerow(
                [
                    vs_row["name"],
                    f"{vs_row.get('dice_pct', ''):+.2f}%",
                    f"{vs_row.get('iou_pct', ''):+.2f}%" if "iou_pct" in vs_row else "",
                    f"{vs_row.get('precision_pct', ''):+.2f}%" if "precision_pct" in vs_row else "",
                    f"{vs_row.get('recall_pct', ''):+.2f}%" if "recall_pct" in vs_row else "",
                    f"{vs_row.get('boundary_f1_pct', ''):+.2f}%" if "boundary_f1_pct" in vs_row else "",
                    f"{vs_row.get('hd95_pct', ''):+.2f}%" if "hd95_pct" in vs_row else "",
                    "derived",
                ]
            )

    md = [
        "## Table 9: Segmentation performance comparison",
        "",
        "| Model | Dice ↑ | IoU ↑ | Precision | Recall | Boundary F1 | HD95 (mm) ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(x: Any, nd: int = 3) -> str:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "—"
        return f"{float(x):.{nd}f}"

    for r in table_rows:
        if r.get("source") == "missing":
            md.append(f"| {r['name']} | — | — | — | — | — | — |")
            continue
        md.append(
            f"| {r['name']} | {fmt(r.get('dice'))} | {fmt(r.get('iou'))} | "
            f"{fmt(r.get('precision'))} | {fmt(r.get('recall'))} | {fmt(r.get('boundary_f1'))} | {fmt(r.get('hd95_mm'), 2)} |"
        )
    if vs_row.get("dice_pct") is not None:
        def pct(v: Any) -> str:
            return f"{v:+.1f}%" if v is not None and isinstance(v, (int, float)) and np.isfinite(v) else "—"

        md.append(
            f"| **{vs_row['name']}** | **{pct(vs_row.get('dice_pct'))}** | "
            f"**{pct(vs_row.get('iou_pct'))}** | **{pct(vs_row.get('precision_pct'))}** | "
            f"**{pct(vs_row.get('recall_pct'))}** | **{pct(vs_row.get('boundary_f1_pct'))}** | "
            f"**{pct(vs_row.get('hd95_pct'))}** |"
        )
    md.append("")
    if dice_ci_primary:
        md.append(f"_95% CI for M-ACAM Dice: [{dice_ci_primary[0]:.3f}, {dice_ci_primary[1]:.3f}]_")
    if "wilcoxon_dice_p" in stats_block:
        md.append(f"_Wilcoxon signed-rank p (Dice vs {ref_name}): {stats_block['wilcoxon_dice_p']:.4g}_")
    if "paired_t_dice" in stats_block:
        t = stats_block["paired_t_dice"]
        md.append(f"_Paired t-test: t={t['statistic']:.2f}, p={t['pvalue']:.4g}, Cohen's d={stats_block.get('cohens_d', float('nan')):.2f}_")
    if stats_block.get("paired_stats_note"):
        md.append(f"_{stats_block['paired_stats_note']}_")

    (out_dir / "table9_segmentation.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(payload, indent=2, default=float))
    print(f"Wrote {out_dir / 'table9_segmentation.json'}")


if __name__ == "__main__":
    main()
