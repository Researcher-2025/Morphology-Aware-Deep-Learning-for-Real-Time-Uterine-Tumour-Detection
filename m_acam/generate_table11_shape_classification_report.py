"""
Table 11: Binary shape classification (Round vs Elongated) + confusion matrix figure.

- static: load counts, optional 2x2 confusion_matrix, and metrics from JSON; write table + PNG.
- measure: run MACAM on test split, collect shape head predictions vs labels (lobulated excluded).

Config:
  m_acam/paper_tables/table11_shape.example.json
  m_acam/paper_tables/table11_shape.measure.example.json

Usage:
  python -m m_acam.generate_table11_shape_classification_report --config m_acam/paper_tables/table11_shape.example.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from m_acam.dataset import UterineFibroidDataset, collate_fn
from m_acam.generate_table8_segmentation_report import load_metadata
from m_acam.model import MACAM
from m_acam.utils import set_global_seed, to_device


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_confusion_matrix_png(
    cm: np.ndarray,
    out_path: Path,
    class_names: Tuple[str, str] = ("Round", "Elongated"),
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    n = len(class_names)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([f"Predicted {c}" for c in class_names], rotation=45, ha="right")
    ax.set_yticklabels([f"Actual {c}" for c in class_names])
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    vmax = float(cm.max()) if cm.size else 1.0
    thresh = vmax / 2.0 if vmax > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _morphology_to_binary_label(morph: Optional[str]) -> Optional[int]:
    if morph == "round":
        return 0
    if morph == "elongated":
        return 1
    return None


def collect_predictions_measure(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds_root = cfg.get("dataset_root", args.dataset_root)
    val_ratio = float(cfg.get("val_ratio", args.val_ratio))
    test_ratio = float(cfg.get("test_ratio", args.test_ratio))
    batch_size = int(cfg.get("batch_size", args.batch_size))
    ckpt_path = cfg.get("checkpoint") or args.checkpoint
    meta_path = (cfg.get("metadata_csv") or "").strip()
    binary_mode = str(cfg.get("binary_pred", "two_logits")).lower()

    if not ckpt_path:
        raise SystemExit("measure mode requires checkpoint in config or --checkpoint")

    meta: Optional[Dict[str, Dict]] = load_metadata(meta_path) if meta_path else None

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
    w = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(w["model"], strict=False)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Shape eval", leave=False):
            b = to_device(batch, device)
            logits = model(b["image"])["shape_logits"]
            if binary_mode == "two_logits":
                l2 = logits[:, :2]
                pred = torch.argmax(l2, dim=1)
                prob_elong = F.softmax(l2, dim=1)[:, 1]
            else:
                pred = torch.argmax(logits, dim=1)
                pred = torch.where(pred == 2, torch.ones_like(pred), pred)
                prob_elong = F.softmax(logits[:, :2], dim=1)[:, 1]

            for i in range(logits.shape[0]):
                iid = batch["image_id"][i]
                if meta is not None:
                    lab = _morphology_to_binary_label(meta.get(iid, {}).get("morphology"))
                else:
                    sl = int(b["shape_label"][i].item())
                    lab = 0 if sl == 0 else (1 if sl == 1 else None)
                if lab is None:
                    continue
                p = int(pred[i].item())
                if p not in (0, 1):
                    p = 1 if p == 2 else int(torch.clamp(torch.tensor(p), 0, 1).item())
                y_true.append(lab)
                y_pred.append(p)
                y_score.append(float(prob_elong[i].item()))

    if not y_true:
        raise SystemExit("No Round/Elongated samples after filtering; check metadata_csv or dataset shape labels.")

    return (
        np.array(y_true, dtype=np.int64),
        np.array(y_pred, dtype=np.int64),
        np.array(y_score, dtype=np.float64),
    )


def labels_from_confusion_matrix(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    yt: List[int] = []
    yp: List[int] = []
    for t in (0, 1):
        for p in (0, 1):
            n = int(cm[t, p])
            yt.extend([t] * n)
            yp.extend([p] * n)
    return np.array(yt, dtype=np.int64), np.array(yp, dtype=np.int64)


def build_metrics_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    labels: Tuple[int, int] = (0, 1),
) -> Tuple[Dict[str, Any], np.ndarray, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    n_round = int((y_true == 0).sum())
    n_elong = int((y_true == 1).sum())

    prec = precision_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)

    mask0 = y_true == 0
    mask1 = y_true == 1
    acc_round = float((y_pred[mask0] == y_true[mask0]).mean()) if mask0.any() else float("nan")
    acc_elong = float((y_pred[mask1] == y_true[mask1]).mean()) if mask1.any() else float("nan")
    overall_acc = float(accuracy_score(y_true, y_pred))
    overall_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    overall_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    overall_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = float("nan")

    metrics = {
        "round": {
            "accuracy": acc_round,
            "precision": float(prec[0]),
            "recall": float(rec[0]),
            "f1": float(f1[0]),
        },
        "elongated": {
            "accuracy": acc_elong,
            "precision": float(prec[1]),
            "recall": float(rec[1]),
            "f1": float(f1[1]),
        },
        "overall": {
            "accuracy": overall_acc,
            "precision": overall_prec,
            "recall": overall_rec,
            "f1": overall_f1,
            "auc_roc": auc,
        },
    }
    return metrics, cm, n_round, n_elong


def pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100.0 * x:.1f}%"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Table 11 shape classification + confusion matrix")
    p.add_argument("--config", type=str, default="m_acam/paper_tables/table11_shape.example.json")
    p.add_argument("--dataset-root", type=str, default="Dataset")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="checkpoints/table11_report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(42)
    cv2.setNumThreads(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    mode = str(cfg.get("mode", "static")).lower()
    names = tuple(cfg.get("class_names", ["Round", "Elongated"]))
    if len(names) != 2:
        names = ("Round", "Elongated")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "table11_confusion_matrix.png"

    if mode == "measure":
        yt, yp, ys = collect_predictions_measure(cfg, args, device)
        metrics_block, cm, n_round, n_elong = build_metrics_payload(yt, yp, ys)
        cm_list = cm.tolist()
        notes = {"mode": "measure", "n_evaluated": int(len(yt))}
    else:
        cm_list = cfg.get("confusion_matrix")
        if cm_list is None:
            raise SystemExit(
                "static mode requires confusion_matrix [[a,b],[c,d]] with rows=true Round, Elongated and cols=pred Round, Elongated"
            )
        cm = np.array(cm_list, dtype=np.int64)
        if cm.shape != (2, 2):
            raise SystemExit("confusion_matrix must be 2x2")
        n_round = int(cfg.get("n_round", int(cm.sum(axis=1)[0])))
        n_elong = int(cfg.get("n_elongated", int(cm.sum(axis=1)[1])))
        metrics_block = cfg.get("metrics")
        if not metrics_block:
            yt_arr, yp_arr = labels_from_confusion_matrix(cm)
            scores = np.zeros(len(yt_arr), dtype=np.float64)
            metrics_block, _, _, _ = build_metrics_payload(yt_arr, yp_arr, scores)
            auc_override = cfg.get("overall_auc_roc")
            if auc_override is not None and str(auc_override) != "":
                metrics_block["overall"]["auc_roc"] = float(auc_override)
        notes = {"mode": "static"}

    plot_confusion_matrix_png(cm, png_path, class_names=(str(names[0]), str(names[1])))

    round_m = metrics_block["round"]
    elong_m = metrics_block["elongated"]
    over_m = metrics_block["overall"]

    payload: Dict[str, Any] = {
        "table": "Table 11: Shape Classification Performance",
        "class_names": list(names),
        "confusion_matrix": cm_list,
        "n_round": n_round,
        "n_elongated": n_elong,
        "metrics": metrics_block,
        "figure": str(png_path.as_posix()),
        "notes": notes,
    }

    def json_default(o: Any) -> Any:
        if isinstance(o, float) and (not np.isfinite(o)):
            return None
        raise TypeError

    (out_dir / "table11_shape.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")

    with open(out_dir / "table11_shape.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "round", "elongated", "overall"])
        w.writerow(["accuracy", round_m["accuracy"], elong_m["accuracy"], over_m["accuracy"]])
        w.writerow(["precision", round_m["precision"], elong_m["precision"], over_m["precision"]])
        w.writerow(["recall", round_m["recall"], elong_m["recall"], over_m["recall"]])
        w.writerow(["f1", round_m["f1"], elong_m["f1"], over_m["f1"]])
        w.writerow(["auc_roc", "", "", over_m.get("auc_roc", "")])

    md = [
        "## Table 11: Shape Classification Performance",
        "",
        f"| Metric | {names[0]} (n={n_round}) | {names[1]} (n={n_elong}) | Overall |",
        "|---|---:|---:|---:|",
        f"| Accuracy | {pct(round_m['accuracy'])} | {pct(elong_m['accuracy'])} | {pct(over_m['accuracy'])} |",
        f"| Precision | {pct(round_m['precision'])} | {pct(elong_m['precision'])} | {pct(over_m['precision'])} |",
        f"| Recall | {pct(round_m['recall'])} | {pct(elong_m['recall'])} | {pct(over_m['recall'])} |",
        f"| F1-Score | {pct(round_m['f1'])} | {pct(elong_m['f1'])} | {pct(over_m['f1'])} |",
    ]
    auc = over_m.get("auc_roc")
    if auc is not None and np.isfinite(float(auc)):
        md.append(f"| AUC-ROC | — | — | {float(auc):.3f} |")
    else:
        md.append("| AUC-ROC | — | — | — |")
    md.extend(["", f"![Confusion matrix]({png_path.name})", ""])

    (out_dir / "table11_shape.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(payload, indent=2, default=json_default))
    print(f"Wrote {out_dir / 'table11_shape.json'}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
