#!/usr/bin/env python3
"""Compare smoothed training loss curves from two vis_data JSON-lines files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal


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


def choose_legend_loc(
    xs_a: list[float], ys_a: list[float], xs_b: list[float], ys_b: list[float]
) -> Literal["upper left", "upper right", "lower left", "lower right"]:
    """Pick the sparsest plot corner for an in-axes legend."""
    xs = xs_a + xs_b
    ys = ys_a + ys_b
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)

    boxes: dict[str, tuple[float, float, float, float]] = {
        "upper left": (0.02, 0.72, 0.44, 0.98),
        "upper right": (0.56, 0.72, 0.98, 0.98),
        "lower left": (0.02, 0.02, 0.44, 0.28),
        "lower right": (0.56, 0.02, 0.98, 0.28),
    }
    scores: dict[str, float] = {}
    for loc, (x0, y0, x1, y1) in boxes.items():
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        score = 0.0
        for x, y in zip(xs, ys):
            xn = (x - x_min) / x_span
            yn = (y - y_min) / y_span
            inside = x0 <= xn <= x1 and y0 <= yn <= y1
            dist2 = (xn - cx) ** 2 + (yn - cy) ** 2
            score += (1000.0 if inside else 0.0) + 1.0 / (dist2 + 0.01)
        scores[loc] = score
    return min(scores, key=scores.get)  # type: ignore[return-value]


def main() -> None:
    args = parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
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

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=args.dpi)
    colors = ["#0072B2", "#D55E00"]
    ax.plot(xs_a, ys_a, linewidth=2.0, color=colors[0], label=args.label_a)
    ax.plot(xs_b, ys_b, linewidth=2.0, color=colors[1], label=args.label_b)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_xlim(min_iter, max_iter)
    ax.grid(True, color="#D9D9D9", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    legend_loc = choose_legend_loc(xs_a, ys_a, xs_b, ys_b)
    legend = ax.legend(
        frameon=True,
        loc=legend_loc,
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=2.0,
        handletextpad=0.5,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#BDBDBD")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_alpha(0.88)
    ax.margins(x=0)
    fig.tight_layout(pad=0.35)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved plot -> {args.output}")
    print(f"Truncated to max iteration -> {int(max_iter)}")


if __name__ == "__main__":
    main()
