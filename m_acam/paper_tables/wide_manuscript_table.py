"""Shared multi-section manuscript tables (Table 15 style, reusable for Table 16+)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

DEFAULT_METRIC_KEYS = (
    "map50",
    "dice",
    "iou",
    "shape_acc",
    "boundary_f1",
    "params_m",
    "params_delta_m",
    "fps",
)

DEFAULT_COLUMN_LABELS: Dict[str, str] = {
    "variant": "Configuration / Variant",
    "map50": "mAP@0.5 ↑",
    "dice": "Dice ↑",
    "iou": "IoU ↑",
    "shape_acc": "Shape Acc ↑",
    "boundary_f1": "Boundary F1 ↑",
    "params_m": "Params (M)",
    "params_delta_m": "ΔParams (M)",
    "fps": "FPS ↑",
    "insight": "Key insights",
}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def section_columns(rows: List[Dict[str, Any]], metric_keys: Sequence[str]) -> List[str]:
    skip = {"insight", "highlight", "significant", "boundary_f1_note"}
    present: Set[str] = set()
    for r in rows:
        for k, v in r.items():
            if k in skip:
                continue
            if v is not None and v != "":
                present.add(k)
    order = ["variant"]
    for k in metric_keys:
        if k in present:
            order.append(k)
    for k in sorted(present):
        if k not in order:
            order.append(k)
    if any("insight" in r for r in rows):
        if "insight" not in order:
            order.append("insight")
    return order


def md_escape(s: str) -> str:
    return str(s).replace("|", "\\|")


def row_cells(r: Dict[str, Any], cols: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in cols:
        if c == "insight":
            out.append(md_escape(str(r.get(c, ""))))
            continue
        if c == "boundary_f1" and r.get("boundary_f1_note"):
            v = r.get(c)
            note = str(r.get("boundary_f1_note", ""))
            if v is None:
                out.append("—")
            else:
                out.append(f"{float(v):.3f} ({note})")
            continue
        v = r.get(c)
        if v is None or v == "":
            out.append("—")
        elif isinstance(v, str):
            out.append(md_escape(v))
        elif isinstance(v, float):
            if c in ("params_m", "params_delta_m"):
                out.append(f"{v:.1f}" if c == "params_m" else f"{v:.2f}")
            elif c == "fps":
                out.append(str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.1f}")
            elif c.endswith("_ms") or c.endswith("_mb"):
                out.append(f"{v:.1f}")
            elif c == "map50":
                out.append(f"{v:.3f}")
            else:
                out.append(f"{v:.3f}")
        elif isinstance(v, int):
            out.append(str(v))
        else:
            out.append(str(v))
    return out


def section_to_markdown(
    sec: Dict[str, Any],
    metric_keys: Sequence[str],
    column_labels: Dict[str, str],
) -> List[str]:
    title = sec.get("title", sec.get("id", ""))
    rows = sec.get("rows") or []
    if not rows:
        return [f"### {title}", "", "_No rows._", ""]
    cols = section_columns(rows, metric_keys)
    lines = [f"### {title}", ""]
    hdr = [column_labels.get(c, c.replace("_", " ").title()) for c in cols]
    seps = []
    for c in cols:
        if c in ("variant", "insight"):
            seps.append(":---")
        else:
            seps.append("---:")
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("| " + " | ".join(seps) + " |")
    for r in rows:
        cells = row_cells(r, cols)
        if r.get("highlight"):
            cells = [f"**{c}**" if c != "—" else c for c in cells]
        if r.get("significant") and "insight" in cols:
            ii = cols.index("insight")
            if cells[ii] and "*" not in cells[ii]:
                cells[ii] = f"{cells[ii]}*"
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def flatten_for_csv(cfg: Dict[str, Any], metric_keys: Sequence[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    fieldnames: Set[str] = {"section_id", "section_title", "variant", "insight"}
    for sec in cfg.get("sections", []):
        for r in sec.get("rows") or []:
            for k in r:
                if k in ("highlight", "significant"):
                    continue
                fieldnames.add(k)
    ordered = ["section_id", "section_title", "variant"]
    for k in metric_keys:
        if k in fieldnames:
            ordered.append(k)
    if "boundary_f1_note" in fieldnames:
        ordered.append("boundary_f1_note")
    for k in sorted(fieldnames):
        if k not in ordered:
            ordered.append(k)
    if "insight" in ordered:
        ordered.remove("insight")
        ordered.append("insight")

    out_rows: List[Dict[str, Any]] = []
    for sec in cfg.get("sections", []):
        sid = sec.get("id", "")
        stitle = sec.get("title", "")
        for r in sec.get("rows") or []:
            row: Dict[str, Any] = {"section_id": sid, "section_title": stitle}
            for k, v in r.items():
                if k in ("highlight", "significant"):
                    continue
                row[k] = v
            out_rows.append(row)
    return ordered, out_rows


def write_manuscript_table_files(cfg: Dict[str, Any], out_dir: Path, basename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = tuple(cfg.get("metric_keys", DEFAULT_METRIC_KEYS))
    column_labels = {**DEFAULT_COLUMN_LABELS, **cfg.get("column_labels", {})}

    (out_dir / f"{basename}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    md_lines = [f"## {cfg.get('title', 'Manuscript table')}", ""]
    for sec in cfg.get("sections", []):
        md_lines.extend(section_to_markdown(sec, metric_keys, column_labels))
    for foot in cfg.get("footnotes", []):
        md_lines.append(f"_{foot}_")
    (out_dir / f"{basename}.md").write_text("\n".join(md_lines), encoding="utf-8")

    fieldnames, csv_rows = flatten_for_csv(cfg, metric_keys)
    with open(out_dir / f"{basename}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in csv_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
