"""
Table 16: Multi-section manuscript table (same JSON schema as Table 15).

Use for robustness, resolution, hardware, quantization, or any tabular addendum.
Optional root keys:
  - metric_keys: column order for numeric fields (default: Table 15 set)
  - column_labels: Markdown header overrides

Config: m_acam/paper_tables/table16_manuscript.example.json

Usage:
  python -m m_acam.generate_table16_report --config m_acam/paper_tables/table16_manuscript.example.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from m_acam.paper_tables.wide_manuscript_table import load_json, write_manuscript_table_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Table 16 manuscript table")
    p.add_argument("--config", type=str, default="m_acam/paper_tables/table16_manuscript.example.json")
    p.add_argument("--out-dir", type=str, default="checkpoints/table16_report")
    p.add_argument("--basename", type=str, default="table16_manuscript", help="Output file stem (no extension)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    out_dir = Path(args.out_dir)
    write_manuscript_table_files(cfg, out_dir, args.basename)
    print(f"Wrote {out_dir / (args.basename + '.md')}")


if __name__ == "__main__":
    main()
