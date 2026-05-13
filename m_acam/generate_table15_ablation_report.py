"""
Table 15: Comprehensive ablation study (multi-section manuscript table).

Reads structured JSON (sections with rows of metric fields + variant + insight).
Writes table15_ablation.json, table15_ablation.csv, table15_ablation.md.

No model inference here — populate JSON from experiments or static paper values.

Usage:
  python -m m_acam.generate_table15_ablation_report --config m_acam/paper_tables/table15_ablation.example.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from m_acam.paper_tables.wide_manuscript_table import load_json, write_manuscript_table_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Table 15 ablation report")
    p.add_argument("--config", type=str, default="m_acam/paper_tables/table15_ablation.example.json")
    p.add_argument("--out-dir", type=str, default="checkpoints/table15_report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    out_dir = Path(args.out_dir)
    write_manuscript_table_files(cfg, out_dir, "table15_ablation")
    print(f"Wrote {out_dir / 'table15_ablation.json'}")
    print(f"Wrote {out_dir / 'table15_ablation.md'}")
    print(f"Wrote {out_dir / 'table15_ablation.csv'}")


if __name__ == "__main__":
    main()
