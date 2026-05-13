"""
Table 12: Size estimation (MAE, RMSE, r, within thresholds) + Bland-Altman summary/plot.
         Eccentricity regression + shape designation agreement.
Table 13: FIGO classification by clinical strata (0-2, 3-4, 5-7) with weighted averages.

Modes per table block: "static" (JSON paper numbers) or "measure" (MACAM on test split).

Optional metadata_csv (Table 8 loader): per-image size_cm, figo overrides when present.

Outputs (under --out-dir):
  table12_clinical.json, .md, .csv
  table12_bland_altman.png (measure mode when size pairs available)
  table13_figo.json, .md, .csv

Usage:
  python -m m_acam.generate_table12_13_reports --config m_acam/paper_tables/table12_13.example.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from m_acam.dataset import UterineFibroidDataset, collate_fn
from m_acam.generate_table8_segmentation_report import load_metadata
from m_acam.model import MACAM
from m_acam.utils import set_global_seed, to_device


FIGO_STRATA: List[Tuple[str, List[int]]] = [
    ("Type 0-2 (Submucosal)", [0, 1, 2]),
    ("Type 3-4 (Intramural)", [3, 4]),
    ("Type 5-7 (Subserosal)", [5, 6, 7]),
]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bootstrap_std_mae_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float]:
    n = len(y_true)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    maes, rmses = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = y_true[idx] - y_pred[idx]
        maes.append(float(np.mean(np.abs(d))))
        rmses.append(float(np.sqrt(np.mean(d**2))))
    return float(np.std(maes)), float(np.std(rmses))


def bootstrap_std_vec(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    return bootstrap_std_mae_rmse(y_true, y_pred, n_boot=n_boot, seed=seed)


def bland_altman_limits(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    diff = pred - true
    mean_d = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    lo = mean_d - 1.96 * sd
    hi = mean_d + 1.96 * sd
    return {"mean_diff": mean_d, "loa_low": float(lo), "loa_high": float(hi)}


def plot_bland_altman(pred: np.ndarray, true: np.ndarray, out_path: Path) -> Dict[str, float]:
    mean_v = (pred + true) / 2.0
    diff = pred - true
    ba = bland_altman_limits(pred, true)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(mean_v, diff, alpha=0.65, s=22, edgecolors="k", linewidths=0.3)
    ax.axhline(ba["mean_diff"], color="C0", linestyle="-", label=f"Mean diff {ba['mean_diff']:.2f} cm")
    ax.axhline(ba["loa_low"], color="C1", linestyle="--", label=f"95% LoA [{ba['loa_low']:.2f}, {ba['loa_high']:.2f}]")
    ax.axhline(ba["loa_high"], color="C1", linestyle="--")
    ax.set_xlabel("Mean of predicted and true size (cm)")
    ax.set_ylabel("Predicted − true (cm)")
    ax.set_title("Bland-Altman (size, cm)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ba


def collect_measure_arrays(
    eval_cfg: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    ds_root = eval_cfg.get("dataset_root", args.dataset_root)
    val_ratio = float(eval_cfg.get("val_ratio", args.val_ratio))
    test_ratio = float(eval_cfg.get("test_ratio", args.test_ratio))
    batch_size = int(eval_cfg.get("batch_size", args.batch_size))
    ckpt = eval_cfg.get("checkpoint") or args.checkpoint
    meta_path = (eval_cfg.get("metadata_csv") or "").strip()
    min_size = float(eval_cfg.get("min_size_cm", 0.01))
    n_boot = int(eval_cfg.get("bootstrap_std_samples", 2000))

    if not ckpt:
        raise SystemExit("measure mode requires table12/table13 checkpoint or --checkpoint")

    meta = load_metadata(meta_path) if meta_path else {}

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

    model = MACAM(pretrained=False, num_det_classes=1).to(device)
    w = torch.load(ckpt, map_location=device)
    model.load_state_dict(w["model"], strict=False)
    model.eval()

    sizes_t: List[float] = []
    sizes_p: List[float] = []
    ecc_t: List[float] = []
    ecc_p: List[float] = []
    shape_true: List[int] = []
    shape_pred: List[int] = []
    figo_t: List[int] = []
    figo_p: List[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Table12/13 eval", leave=False):
            b = to_device(batch, device)
            out = model(b["image"])
            sp = out["size_pred"].squeeze(-1).cpu().numpy()
            ep = out["ecc_pred"].squeeze(-1).cpu().numpy()
            sl = out["shape_logits"]
            pred_shape = torch.argmax(sl[:, :2], dim=1).cpu().numpy()
            pred_figo = torch.argmax(out["figo_logits"], dim=1).cpu().numpy()

            sz_b = b["size_cm"].cpu().numpy()
            ecc_b = b["eccentricity"].cpu().numpy()
            sh_b = b["shape_label"].cpu().numpy().astype(np.int64)
            fg_b = b["figo_label"].cpu().numpy().astype(np.int64)

            for i in range(sp.shape[0]):
                iid = batch["image_id"][i]
                m = meta.get(iid, {}) if meta else {}
                st = float(m.get("size_cm", sz_b[i]) if m.get("size_cm") is not None else sz_b[i])
                if not np.isfinite(st) or st <= min_size:
                    pass
                else:
                    sizes_t.append(st)
                    sizes_p.append(float(sp[i]))
                ecc_t.append(float(ecc_b[i]))
                ecc_p.append(float(ep[i]))
                if int(sh_b[i]) in (0, 1):
                    shape_true.append(int(sh_b[i]))
                    shape_pred.append(int(pred_shape[i]))
                ft = m.get("figo")
                if ft is not None and str(ft).strip() != "":
                    try:
                        figo_t.append(int(ft))
                    except Exception:
                        figo_t.append(int(fg_b[i]))
                else:
                    figo_t.append(int(fg_b[i]))
                figo_p.append(int(pred_figo[i]))

    out_d: Dict[str, Any] = {
        "size_true": np.array(sizes_t, dtype=np.float64),
        "size_pred": np.array(sizes_p, dtype=np.float64),
        "ecc_true": np.array(ecc_t, dtype=np.float64),
        "ecc_pred": np.array(ecc_p, dtype=np.float64),
        "shape_true": np.array(shape_true, dtype=np.int64),
        "shape_pred": np.array(shape_pred, dtype=np.int64),
        "figo_true": np.array(figo_t, dtype=np.int64),
        "figo_pred": np.array(figo_p, dtype=np.int64),
        "n_boot": n_boot,
    }
    return out_d


def build_table12_measure(cfg12: Dict[str, Any], arrs: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    st, sp = arrs["size_true"], arrs["size_pred"]
    n_boot = int(arrs.get("n_boot", 2000))
    size_block: Dict[str, Any] = {}
    ba_block: Dict[str, Any] = {}
    ecc_block: Dict[str, Any] = {}

    if len(st) == len(sp) and len(st) > 1:
        err = sp - st
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        try:
            r = float(pearsonr(st, sp)[0])
        except Exception:
            r = float("nan")
        mae_std, rmse_std = bootstrap_std_mae_rmse(st, sp, n_boot=n_boot)
        within05 = float(np.mean(np.abs(err) <= 0.5))
        within10 = float(np.mean(np.abs(err) <= 1.0))
        size_block = {
            "mae_cm": mae,
            "mae_std_cm": mae_std,
            "rmse_cm": rmse,
            "rmse_std_cm": rmse_std,
            "correlation_r": float(r),
            "within_0_5_cm_pct": within05,
            "within_1_0_cm_pct": within10,
            "thresholds": (cfg12.get("size_estimation") or {}).get("thresholds", {}),
        }
        ba_block = bland_altman_limits(sp, st)
    else:
        size_block = {"error": "insufficient_size_pairs", "n_pairs": int(min(len(st), len(sp)))}

    et, ep = arrs["ecc_true"], arrs["ecc_pred"]
    if len(et) > 1:
        e_err = ep - et
        try:
            r_e = float(pearsonr(et, ep)[0])
        except Exception:
            r_e = float("nan")
        ecc_block = {
            "mae": float(np.mean(np.abs(e_err))),
            "mae_std": bootstrap_std_mae_rmse(et, ep, n_boot=n_boot)[0],
            "rmse": float(np.sqrt(np.mean(e_err**2))),
            "rmse_std": bootstrap_std_mae_rmse(et, ep, n_boot=n_boot)[1],
            "correlation_r": r_e,
        }
    else:
        ecc_block = {"error": "insufficient_ecc_samples"}

    sth, spr = arrs["shape_true"], arrs["shape_pred"]
    if len(sth) > 0:
        ecc_block["shape_agreement_pct"] = float(np.mean(sth == spr))
    else:
        ecc_block["shape_agreement_pct"] = float("nan")

    return {"size_estimation": size_block, "bland_altman_cm": ba_block, "eccentricity": ecc_block}, st, sp


def build_table12_static(cfg12: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "size_estimation": cfg12.get("size_estimation", {}),
        "bland_altman_cm": cfg12.get("bland_altman_cm", {}),
        "eccentricity": cfg12.get("eccentricity", {}),
    }


def figo_stratum_stats(y_true: np.ndarray, y_pred: np.ndarray, classes: List[int]) -> Tuple[int, float, float]:
    mask = np.isin(y_true, classes)
    n = int(mask.sum())
    if n == 0:
        return 0, float("nan"), float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    acc = float(accuracy_score(yt, yp))
    f1 = float(f1_score(yt, yp, labels=classes, average="macro", zero_division=0))
    return n, acc, f1


def build_table13_measure(cfg13: Dict[str, Any], arrs: Dict[str, Any]) -> Dict[str, Any]:
    yt, yp = arrs["figo_true"], arrs["figo_pred"]
    valid = (yt >= 0) & (yt <= 7)
    yt = yt[valid]
    yp = yp[valid]
    rows_out: List[Dict[str, Any]] = []
    total_n = 0
    sum_acc_w = 0.0
    sum_f1_w = 0.0

    for name, cls in FIGO_STRATA:
        n, acc, f1 = figo_stratum_stats(yt, yp, cls)
        rows_out.append({"name": name, "figo_classes": cls, "count": n, "accuracy": acc, "f1": f1})
        if n > 0 and np.isfinite(acc) and np.isfinite(f1):
            total_n += n
            sum_acc_w += acc * n
            sum_f1_w += f1 * n

    w_acc = float(sum_acc_w / total_n) if total_n else float("nan")
    w_f1 = float(sum_f1_w / total_n) if total_n else float("nan")
    return {
        "rows": rows_out,
        "weighted_average": {"count": total_n, "accuracy": w_acc, "f1": w_f1},
    }


def build_table13_static(cfg13: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rows": cfg13.get("rows", []),
        "weighted_average": cfg13.get("weighted_average", {}),
    }


def fmt_pct(x: Any, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100.0 * float(x):.{nd}f}%"


def fmt_pm(mean: float, std: float, nd: int = 2) -> str:
    if not np.isfinite(mean):
        return "—"
    if not np.isfinite(std):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def write_table12_md(
    t12: Dict[str, Any],
    bland_png: Optional[str],
    out_md: Path,
) -> None:
    se = t12.get("size_estimation", {})
    ba = t12.get("bland_altman_cm", {})
    ec = t12.get("eccentricity", {})
    th = se.get("thresholds", {})

    lines = [
        "## Table 12: Size Estimation Accuracy",
        "",
        "| Metric | Value | Clinical threshold |",
        "|---|---|---|",
    ]
    if "mae_cm" in se:
        mae_s = th.get("mae_accept_cm", "< 0.5 cm (acceptable)")
        rmse_s = th.get("rmse_accept_cm", "< 0.7 cm (acceptable)")
        r_s = th.get("r_excellent", "> 0.9 (excellent)")
        w05 = th.get("within_0_5_target", "> 80% (target)")
        w10 = th.get("within_1_0_target", "> 95% (target)")
        lines.append(
            f"| MAE | {fmt_pm(se['mae_cm'], se.get('mae_std_cm', float('nan')))} cm | {mae_s} |"
        )
        lines.append(
            f"| RMSE | {fmt_pm(se['rmse_cm'], se.get('rmse_std_cm', float('nan')))} cm | {rmse_s} |"
        )
        r0 = se.get("correlation_r", float("nan"))
        r_txt = f"{float(r0):.3f}" if np.isfinite(float(r0)) else "—"
        lines.append(f"| Correlation (r) | {r_txt} | {r_s} |")
        lines.append(f"| Within 0.5 cm | {fmt_pct(se.get('within_0_5_cm_pct'))} | {w05} |")
        lines.append(f"| Within 1.0 cm | {fmt_pct(se.get('within_1_0_cm_pct'))} | {w10} |")
    else:
        lines.append(f"| (size) | {se.get('error', 'n/a')} | — |")

    lines.extend(
        [
            "",
            "### Supplemental analysis",
            "",
        ]
    )
    if ba:
        md = ba.get("mean_diff", float("nan"))
        lo = ba.get("loa_low", float("nan"))
        hi = ba.get("loa_high", float("nan"))
        if all(np.isfinite(float(x)) for x in (md, lo, hi)):
            lines.append(
                f"- **Bland-Altman:** mean difference **{md:.2f} cm**, "
                f"95% limits **[{lo:.2f}, {hi:.2f}] cm**."
            )
        else:
            lines.append(f"- **Bland-Altman:** {ba}")
    if "mae" in ec and "error" not in ec:
        r1 = ec.get("correlation_r", float("nan"))
        r_txt = f"{float(r1):.3f}" if np.isfinite(float(r1)) else "—"
        lines.extend(
            [
                "- **Eccentricity prediction:**",
                f"  - MAE: {fmt_pm(ec['mae'], ec.get('mae_std', float('nan')))}",
                f"  - RMSE: {fmt_pm(ec['rmse'], ec.get('rmse_std', float('nan')))}",
                f"  - Correlation (r): {r_txt}",
                f"  - Shape designation agreement: {fmt_pct(ec.get('shape_agreement_pct'))}",
            ]
        )
    elif "error" in ec:
        lines.append(f"- **Eccentricity:** {ec.get('error')}")
        sag = ec.get("shape_agreement_pct")
        if sag is not None and np.isfinite(float(sag)):
            lines.append(f"  - Shape designation agreement: {fmt_pct(sag)}")
    if bland_png:
        lines.extend(["", f"![Bland-Altman]({Path(bland_png).name})", ""])
    out_md.write_text("\n".join(lines), encoding="utf-8")


def write_table13_md(t13: Dict[str, Any], out_md: Path) -> None:
    rows = t13.get("rows", [])
    w = t13.get("weighted_average", {})
    lines = [
        "## Table 13: FIGO Classification",
        "",
        "| FIGO Type | Count | Accuracy | F1-Score |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        acc = r.get("accuracy", float("nan"))
        f1 = r.get("f1", float("nan"))
        lines.append(
            f"| {r.get('name', '')} | {r.get('count', 0)} | {fmt_pct(acc)} | "
            f"{(f'{f1:.3f}' if np.isfinite(f1) else '—')} |"
        )
    if w:
        accw = w.get("accuracy", float("nan"))
        f1w = w.get("f1", float("nan"))
        lines.append(
            f"| **Weighted average** | **{w.get('count', 0)}** | **{fmt_pct(accw)}** | "
            f"**{(f'{f1w:.3f}' if np.isfinite(f1w) else '—')}** |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Table 12 and 13 clinical reports")
    p.add_argument("--config", type=str, default="m_acam/paper_tables/table12_13.example.json")
    p.add_argument("--dataset-root", type=str, default="Dataset")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="checkpoints/table12_13_report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(42)
    cv2.setNumThreads(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_json(args.config)
    cfg12 = cfg.get("table12") or {}
    cfg13 = cfg.get("table13") or {}
    mode12 = str(cfg12.get("mode", "static")).lower()
    mode13 = str(cfg13.get("mode", "static")).lower()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bland_path = out_dir / "table12_bland_altman.png"
    bland_rel: Optional[str] = None

    arrs: Optional[Dict[str, Any]] = None
    if mode12 == "measure" or mode13 == "measure":
        if mode12 == "measure" and mode13 == "measure":
            eval_cfg = {**cfg12, **cfg13}
        elif mode12 == "measure":
            eval_cfg = dict(cfg12)
        else:
            eval_cfg = dict(cfg13)
        if not (eval_cfg.get("checkpoint") or args.checkpoint):
            raise SystemExit("measure mode requires checkpoint in config or --checkpoint")
        arrs = collect_measure_arrays(eval_cfg, args, device)

    if mode12 == "measure":
        assert arrs is not None
        t12, st, sp = build_table12_measure(cfg12, arrs)
        if st is not None and sp is not None and len(st) == len(sp) and len(st) > 1:
            plot_bland_altman(sp, st, bland_path)
            bland_rel = str(bland_path.name)
    else:
        t12 = build_table12_static(cfg12)

    if mode13 == "measure":
        assert arrs is not None
        t13 = build_table13_measure(cfg13, arrs)
    else:
        t13 = build_table13_static(cfg13)

    def jdef(o: Any) -> Any:
        if isinstance(o, float) and (not np.isfinite(o)):
            return None
        raise TypeError

    payload = {"table12": t12, "table13": t13}
    (out_dir / "table12_13_combined.json").write_text(json.dumps(payload, indent=2, default=jdef), encoding="utf-8")
    (out_dir / "table12_clinical.json").write_text(json.dumps({"table12": t12}, indent=2, default=jdef), encoding="utf-8")
    (out_dir / "table13_figo.json").write_text(json.dumps({"table13": t13}, indent=2, default=jdef), encoding="utf-8")

    write_table12_md(t12, bland_rel, out_dir / "table12_clinical.md")
    write_table13_md(t13, out_dir / "table13_figo.md")

    with open(out_dir / "table12_clinical.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "note"])
        se = t12.get("size_estimation", {})
        for k in ["mae_cm", "rmse_cm", "correlation_r", "within_0_5_cm_pct", "within_1_0_cm_pct"]:
            if k in se:
                w.writerow([k, se[k], ""])
        for k, v in t12.get("bland_altman_cm", {}).items():
            w.writerow([f"bland_altman_{k}", v, ""])
        for k, v in t12.get("eccentricity", {}).items():
            w.writerow([f"eccentricity_{k}", v, ""])

    with open(out_dir / "table13_figo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["figo_type", "count", "accuracy", "f1"])
        for r in t13.get("rows", []):
            w.writerow([r.get("name"), r.get("count"), r.get("accuracy"), r.get("f1")])
        wa = t13.get("weighted_average", {})
        if wa:
            w.writerow(["Weighted average", wa.get("count"), wa.get("accuracy"), wa.get("f1")])

    print(json.dumps(payload, indent=2, default=jdef))
    print(f"Wrote {out_dir / 'table12_13_combined.json'}")


if __name__ == "__main__":
    main()
