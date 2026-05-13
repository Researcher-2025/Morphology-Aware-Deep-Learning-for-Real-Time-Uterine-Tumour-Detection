"""
Generate Table 8 style segmentation report:
Overall, Morphology, Size, and FIGO-stratified Dice comparison.

Inputs:
- checkpoint for M-ACAM
- checkpoint for baseline model
- metadata CSV with at least: image_id, size_cm, figo
  and either morphology_label OR eccentricity

Outputs:
- table8_segmentation.json
- table8_segmentation.csv
- table8_segmentation.md
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from m_acam.dataset import UterineFibroidDataset, collate_fn
from m_acam.model import MACAM
from m_acam.utils import set_global_seed, to_device

try:
    from scipy.stats import wilcoxon  # type: ignore
except Exception:  # pragma: no cover
    wilcoxon = None


def dice_per_image(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return float((2.0 * inter + eps) / (union + eps))


def bootstrap_ci(values: List[float], n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(float(arr[idx].mean()))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5])
    return float(lo), float(hi)


def load_metadata(path: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            image_id = str(r.get("image_id", "")).strip()
            if not image_id:
                continue

            # morphology label priority: explicit label > shape label > eccentricity threshold
            morphology = str(r.get("morphology_label", "")).strip().lower()
            if not morphology:
                shape = str(r.get("shape", "")).strip().lower()
                if shape in ["round", "elongated", "lobulated"]:
                    morphology = shape
                elif shape.isdigit():
                    shape_i = int(shape)
                    morphology = {0: "round", 1: "elongated", 2: "lobulated"}.get(shape_i, "")
            if not morphology:
                ecc_val = r.get("eccentricity", "")
                if ecc_val not in [None, ""]:
                    ecc = float(ecc_val)
                    morphology = "round" if ecc < 0.7 else "elongated"

            size_cm = float(r.get("size_cm", r.get("diameter_cm", "nan")))
            figo_raw = r.get("figo", r.get("figo_type", ""))
            figo = int(figo_raw) if str(figo_raw).strip().isdigit() else None

            out[image_id] = {
                "morphology": morphology if morphology else None,
                "size_cm": size_cm if np.isfinite(size_cm) else None,
                "figo": figo,
            }
    return out


def size_bucket(size_cm: float) -> str:
    if size_cm < 3.0:
        return "small_lt_3cm"
    if size_cm <= 8.0:
        return "medium_3_8cm"
    return "large_gt_8cm"


def infer_dice_map(model: MACAM, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Seg infer", leave=False):
            b = to_device(batch, device)
            pred = torch.sigmoid(model(b["image"])["seg_logits"])
            for i, image_id in enumerate(batch["image_id"]):
                d = dice_per_image(pred[i : i + 1], b["mask"][i : i + 1])
                out[image_id] = d
    return out


def subgroup_stats(name: str, ids: List[str], dice_a: Dict[str, float], dice_b: Dict[str, float]) -> Dict:
    va = [dice_a[i] for i in ids if i in dice_a and i in dice_b]
    vb = [dice_b[i] for i in ids if i in dice_a and i in dice_b]
    count = len(va)
    if count == 0:
        return {
            "name": name,
            "count": 0,
            "macam_dice": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "baseline_dice": float("nan"),
            "improvement_pct": float("nan"),
            "p_wilcoxon": float("nan"),
        }

    mean_a = float(np.mean(va))
    mean_b = float(np.mean(vb))
    ci = bootstrap_ci(va)
    improve = ((mean_a - mean_b) / max(mean_b, 1e-8)) * 100.0

    if wilcoxon is not None and count > 1:
        try:
            p = float(wilcoxon(va, vb, zero_method="wilcox", alternative="two-sided").pvalue)
        except Exception:
            p = float("nan")
    else:
        p = float("nan")

    return {
        "name": name,
        "count": count,
        "macam_dice": mean_a,
        "ci95": [ci[0], ci[1]],
        "baseline_dice": mean_b,
        "improvement_pct": improve,
        "p_wilcoxon": p,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate Table 8 segmentation report")
    p.add_argument("--dataset-root", type=str, default="Dataset")
    p.add_argument("--checkpoint-macam", type=str, required=True)
    p.add_argument("--checkpoint-baseline", type=str, required=True)
    p.add_argument("--baseline-name", type=str, default="Baseline")
    p.add_argument("--metadata-csv", type=str, required=True, help="CSV with image_id + morphology/size/figo fields")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="checkpoints/table8_report")
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

    m1 = MACAM(pretrained=False, num_det_classes=1).to(device)
    m2 = MACAM(pretrained=False, num_det_classes=1).to(device)
    ck1 = torch.load(args.checkpoint_macam, map_location=device)
    ck2 = torch.load(args.checkpoint_baseline, map_location=device)
    m1.load_state_dict(ck1["model"], strict=False)
    m2.load_state_dict(ck2["model"], strict=False)

    dice_macam = infer_dice_map(m1, loader, device)
    dice_base = infer_dice_map(m2, loader, device)
    meta = load_metadata(args.metadata_csv)

    ids_all = [iid for iid in dice_macam.keys() if iid in dice_base]
    overall = subgroup_stats("overall", ids_all, dice_macam, dice_base)

    ids_round = [i for i in ids_all if meta.get(i, {}).get("morphology") == "round"]
    ids_elong = [i for i in ids_all if meta.get(i, {}).get("morphology") == "elongated"]
    ids_lob = [i for i in ids_all if meta.get(i, {}).get("morphology") == "lobulated"]

    ids_small = [i for i in ids_all if meta.get(i, {}).get("size_cm") is not None and size_bucket(meta[i]["size_cm"]) == "small_lt_3cm"]
    ids_medium = [i for i in ids_all if meta.get(i, {}).get("size_cm") is not None and size_bucket(meta[i]["size_cm"]) == "medium_3_8cm"]
    ids_large = [i for i in ids_all if meta.get(i, {}).get("size_cm") is not None and size_bucket(meta[i]["size_cm"]) == "large_gt_8cm"]

    figo_groups: Dict[str, List[str]] = {}
    for t in range(8):
        figo_groups[f"figo_type_{t}"] = [i for i in ids_all if meta.get(i, {}).get("figo") == t]

    rows = [
        overall,
        subgroup_stats("morphology_round", ids_round, dice_macam, dice_base),
        subgroup_stats("morphology_elongated", ids_elong, dice_macam, dice_base),
        subgroup_stats("morphology_lobulated", ids_lob, dice_macam, dice_base),
        subgroup_stats("size_small_lt_3cm", ids_small, dice_macam, dice_base),
        subgroup_stats("size_medium_3_8cm", ids_medium, dice_macam, dice_base),
        subgroup_stats("size_large_gt_8cm", ids_large, dice_macam, dice_base),
    ]
    for k in sorted(figo_groups.keys()):
        rows.append(subgroup_stats(k, figo_groups[k], dice_macam, dice_base))

    payload = {
        "baseline_name": args.baseline_name,
        "rows": rows,
        "notes": {
            "test_count": len(ids_all),
            "significance_test": "Wilcoxon signed-rank (requires scipy)",
            "scipy_available": wilcoxon is not None,
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table8_segmentation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with open(out_dir / "table8_segmentation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "category",
                "count",
                "macam_dice",
                "ci95_lo",
                "ci95_hi",
                "baseline_dice",
                "improvement_pct",
                "p_wilcoxon",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["name"],
                    r["count"],
                    r["macam_dice"],
                    r["ci95"][0],
                    r["ci95"][1],
                    r["baseline_dice"],
                    r["improvement_pct"],
                    r["p_wilcoxon"],
                ]
            )

    md = [
        "## Table 8 Style Segmentation Report",
        "",
        f"Baseline: **{args.baseline_name}**",
        "",
        "| Category | Count | M-ACAM Dice | 95% CI | Baseline Dice | Improvement vs Baseline | p-value (Wilcoxon) |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        ci = f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]" if np.isfinite(r["ci95"][0]) else "-"
        md.append(
            f"| {r['name']} | {r['count']} | {r['macam_dice']:.3f} | {ci} | {r['baseline_dice']:.3f} | {r['improvement_pct']:+.1f}% | {r['p_wilcoxon']:.4g} |"
        )
    (out_dir / "table8_segmentation.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_dir / 'table8_segmentation.json'}")
    print(f"Wrote {out_dir / 'table8_segmentation.csv'}")
    print(f"Wrote {out_dir / 'table8_segmentation.md'}")


if __name__ == "__main__":
    main()

