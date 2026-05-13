"""
Table 17: Demographic / clinical subgroup fairness (Dice, n, Δ vs overall).

Static config only — populate from stratified evaluation or manuscript numbers.
Optional row flag `significant: true` appends * to the Δ vs Overall cell in Markdown.

Config: m_acam/paper_tables/table17_fairness.example.json

Usage:
  python -m m_acam.generate_table17_fairness_report --config m_acam/paper_tables/table17_fairness.example.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_delta(pct: Any, significant: bool) -> str:
    if pct is None:
        return "—"
    v = float(pct)
    s = f"{v:+.1f}%"
    return f"{s}*" if significant else s


def build_markdown(cfg: Dict[str, Any]) -> str:
    lines: List[str] = []
    sec = cfg.get("section_title", "")
    if sec:
        lines.append(f"## {sec}")
        lines.append("")
    intro = cfg.get("intro_paragraph", "")
    if intro:
        lines.append(intro)
        lines.append("")
    lines.append(f"### {cfg.get('table_title', 'Table 17')}")
    lines.append("")
    lines.append("| Subgroup | n | Dice | Δ vs Overall |")
    lines.append("|:---|---:|---:|---:|")
    for r in cfg.get("rows", []):
        sig = bool(r.get("significant"))
        delta = fmt_delta(r.get("delta_vs_overall_pct"), sig)
        dice = r.get("dice")
        dice_s = f"{float(dice):.3f}" if dice is not None else "—"
        lines.append(f"| {r.get('subgroup', '')} | {int(r.get('n', 0))} | {dice_s} | {delta} |")
    lines.append("")
    for foot in cfg.get("footnotes", []):
        lines.append(f"_{foot}_")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Table 17 fairness / subgroup Dice report")
    p.add_argument("--config", type=str, default="m_acam/paper_tables/table17_fairness.example.json")
    p.add_argument("--out-dir", type=str, default="checkpoints/table17_report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "table17_fairness.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    with open(out_dir / "table17_fairness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subgroup", "n", "dice", "delta_vs_overall_pct", "significant"])
        for r in cfg.get("rows", []):
            w.writerow(
                [
                    r.get("subgroup", ""),
                    r.get("n", ""),
                    r.get("dice", ""),
                    r.get("delta_vs_overall_pct", ""),
                    r.get("significant", False),
                ]
            )

    (out_dir / "table17_fairness.md").write_text(build_markdown(cfg), encoding="utf-8")
    print(f"Wrote {out_dir / 'table17_fairness.md'}")


if __name__ == "__main__":
    main()
