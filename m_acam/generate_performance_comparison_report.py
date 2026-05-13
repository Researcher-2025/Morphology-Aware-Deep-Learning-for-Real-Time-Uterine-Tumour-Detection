"""
Generate Table 6–style performance comparison: FPS, Memory (MB), Dice, Efficiency.

Efficiency (paper):  (Dice × FPS) / (Memory / 1000)

- Baselines: provide dice, fps, memory_mb in JSON (from paper or your runs).
- M-ACAM: set "measure_runtime": true to benchmark FPS & peak GPU memory on this machine;
  dice can come from your seg evaluation JSON or set manually in the config.

Usage:
  python -m m_acam.generate_performance_comparison_report ^
    --config m_acam/paper_tables/table6_performance.example.json ^
    --out-dir checkpoints/paper_reports

Optional figure (Fig. 4 style):
  python -m m_acam.generate_performance_comparison_report ... --plot figure4.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from m_acam.model import MACAM
from m_acam.paper_tables.runtime_bench import (
    benchmark_fps,
    model_memory_estimate_mb,
    peak_inference_memory_mb_cuda,
)


def efficiency_score(dice: float, fps: float, memory_mb: float) -> float:
    return (dice * fps) / max(memory_mb / 1000.0, 1e-9)


def measure_macam_runtime(
    input_shape: Tuple[int, int, int, int],
    warmup: int,
    repeats: int,
    device: torch.device,
) -> Tuple[float, float, str]:
    model = MACAM(pretrained=False, num_det_classes=1).to(device)
    model.eval()

    fps = benchmark_fps(model, device, input_shape, warmup=warmup, repeats=repeats)

    if device.type == "cuda":
        mem = peak_inference_memory_mb_cuda(model, device, input_shape)
        note = "Peak CUDA memory (MB) during one forward pass including activations."
    else:
        mem = model_memory_estimate_mb(model, trainable_only=False)
        note = "CPU fallback: parameter+buffer size (MB); not peak activation memory."

    return fps, mem, note


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_rows(cfg: Dict[str, Any], device: torch.device) -> List[Dict[str, Any]]:
    inp = tuple(cfg.get("input_size", [1, 1, 512, 512]))
    bench = cfg.get("benchmark", {})
    warmup = int(bench.get("warmup", 10))
    repeats = int(bench.get("repeats", 100))

    out_rows: List[Dict[str, Any]] = []
    for m in cfg["models"]:
        key = m["key"]
        display = m["display_name"]
        dice = float(m["dice"])
        fps: Optional[float] = m.get("fps")
        mem: Optional[float] = m.get("memory_mb")
        note = m.get("note", "")
        measure = bool(m.get("measure_runtime", False))

        runtime_note = ""
        if measure:
            if key != "macam":
                raise ValueError(
                    f"measure_runtime is only wired for key 'macam' in this script; got '{key}'."
                )
            fps_m, mem_m, runtime_note = measure_macam_runtime(inp, warmup, repeats, device)
            fps, mem = fps_m, mem_m
            note = (note + " " + runtime_note).strip()

        if fps is None or mem is None:
            raise ValueError(f"Model '{display}' missing fps/memory and measure_runtime is false.")

        eff = efficiency_score(dice, float(fps), float(mem))
        out_rows.append(
            {
                "key": key,
                "model": display,
                "fps": round(float(fps), 2),
                "memory_mb": round(float(mem), 2),
                "dice": dice,
                "efficiency_score": round(eff, 2),
                "notes": note,
            }
        )

    return out_rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "fps", "memory_mb", "dice", "efficiency_score", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_md(rows: List[Dict[str, Any]], path: Path, formula: str) -> None:
    lines = [
        "## Performance comparison (Table 6 style)",
        "",
        f"_Efficiency score:_ {formula}",
        "",
        "| Model | FPS | Memory (MB) | Dice | Efficiency Score |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['fps']} | {r['memory_mb']} | {r['dice']} | {r['efficiency_score']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def try_plot(rows: List[Dict[str, Any]], out_png: str, highlight: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to use --plot.") from exc

    names = [r["model"] for r in rows]
    eff = [r["efficiency_score"] for r in rows]
    dice = [r["dice"] for r in rows]
    mem = [r["memory_mb"] for r in rows]
    fps = [r["fps"] for r in rows]

    colors = ["#ff7f0e" if highlight.lower() in n.lower() or n == highlight else "#1f77b4" for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Performance Comparison: Proposed M-ACAM vs Baselines")

    def bar(ax, vals, title, ylabel, col):
        ax.bar(names, vals, color=col)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)

    bar(axes[0, 0], eff, "Efficiency Score", "Score", colors)
    bar(axes[0, 1], dice, "Dice Coefficient", "Dice", colors)
    bar(axes[1, 0], mem, "Memory (MB)", "MB", colors)
    bar(axes[1, 1], fps, "FPS", "FPS", colors)

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Performance table (Table 6) + optional Fig. 4 bars")
    p.add_argument("--config", type=str, default="m_acam/paper_tables/table6_performance.example.json")
    p.add_argument("--out-dir", type=str, default="checkpoints/paper_reports")
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu (CPU memory note differs; FPS not comparable to GPU paper).",
    )
    p.add_argument("--plot", type=str, default="", help="Optional PNG path for 4-panel bar figure.")
    p.add_argument("--highlight-model", type=str, default="M-ACAM", help="Substring to color orange in plots.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if args.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available; using CPU for benchmarking.")

    rows = build_rows(cfg, device)
    formula = cfg.get("efficiency_formula", "(dice * fps) / (memory_mb / 1000.0)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "efficiency_formula": formula,
        "device": str(device),
        "input_size": cfg.get("input_size"),
        "benchmark": cfg.get("benchmark"),
        "rows": rows,
    }
    (out_dir / "performance_table6.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(rows, out_dir / "performance_table6.csv")
    write_md(rows, out_dir / "performance_table6.md", formula=f"`{formula}`")

    print(f"Wrote {out_dir / 'performance_table6.json'}")
    print(f"Wrote {out_dir / 'performance_table6.csv'}")
    print(f"Wrote {out_dir / 'performance_table6.md'}")

    if args.plot:
        try_plot(rows, args.plot, args.highlight_model)
        print(f"Wrote plot {args.plot}")


if __name__ == "__main__":
    main()
