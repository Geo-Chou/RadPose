#!/usr/bin/env python3
"""Compare smoothed training loss curves from two vis_data JSON-lines files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("vis_json_a", type=Path)
    parser.add_argument("vis_json_b", type=Path)
    parser.add_argument("--label-a", default="regression without heatmap")
    parser.add_argument("--label-b", default="regression with heatmap")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoothing", type=float, default=0.6)
    parser.add_argument("--scale-a", type=float, default=7.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_loss_series(path: Path) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "iter" not in record or "loss" not in record:
                continue
            xs.append(float(record["iter"]))
            ys.append(float(record["loss"]))
    pairs = sorted(zip(xs, ys), key=lambda pair: pair[0])
    if not pairs:
        return [], []
    xs_sorted, ys_sorted = zip(*pairs)
    return list(xs_sorted), list(ys_sorted)


def ema(ys: list[float], smoothing: float) -> list[float]:
    if not ys:
        return []
    out: list[float] = []
    last = ys[0]
    for y in ys:
        value = smoothing * last + (1 - smoothing) * y
        out.append(value)
        last = value
    return out


def truncate(xs: list[float], ys: list[float], max_iter: float) -> tuple[list[float], list[float]]:
    kept = [(x, y) for x, y in zip(xs, ys) if x <= max_iter]
    if not kept:
        return [], []
    xs_out, ys_out = zip(*kept)
    return list(xs_out), list(ys_out)


def filter_range(
    xs: list[float], ys: list[float], min_iter: float, max_iter: float
) -> tuple[list[float], list[float]]:
    kept = [(x, y) for x, y in zip(xs, ys) if min_iter <= x <= max_iter]
    if not kept:
        return [], []
    xs_out, ys_out = zip(*kept)
    return list(xs_out), list(ys_out)


def main() -> None:
    args = parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
            "axes.labelsize": 36,
            "xtick.labelsize": 36,
            "ytick.labelsize": 36,
            "legend.fontsize": 36,
        }
    )

    xs_a, ys_a = load_loss_series(args.vis_json_a.expanduser().resolve())
    xs_b, ys_b = load_loss_series(args.vis_json_b.expanduser().resolve())
    if not xs_a or not xs_b:
        raise ValueError("Both files must contain iter/loss training records.")

    min_iter = max(xs_a[0], xs_b[0])
    max_iter = min(xs_a[-1], xs_b[-1])
    xs_a, ys_a = filter_range(xs_a, ys_a, min_iter, max_iter)
    xs_b, ys_b = filter_range(xs_b, ys_b, min_iter, max_iter)

    ys_a = [y * args.scale_a for y in ys_a]
    ys_a = ema(ys_a, args.smoothing)
    ys_b = ema(ys_b, args.smoothing)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 8), dpi=args.dpi)
    ax.plot(xs_a, ys_a, linewidth=3.5, label=args.label_a)
    ax.plot(xs_b, ys_b, linewidth=3.5, label=args.label_b)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_xlim(min_iter, max_iter)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.margins(x=0)
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.18, top=0.82)
    fig.savefig(args.output)
    plt.close(fig)

    print(f"Saved plot -> {args.output}")
    print(f"Truncated to max iteration -> {int(max_iter)}")


if __name__ == "__main__":
    main()
