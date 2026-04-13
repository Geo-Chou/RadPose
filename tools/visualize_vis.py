#!/usr/bin/env python3
"""Plot training and evaluation metrics from MMEngine vis_data JSON lines."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

DEFAULT_TRAIN_METRICS = ("loss", "train/loss")
IGNORE_METRICS = {"time", "data_time", "memory", "eta", "lr", "grad_norm", "num_persons" , "joint", "MPJPE"}


@dataclass
class MetricSeries:
    name: str
    xs: List[float] = field(default_factory=list)
    ys: List[float] = field(default_factory=list)

    def add_point(self, x_value: float, y_value: float) -> None:
        self.xs.append(x_value)
        self.ys.append(y_value)

    def sorted_points(self) -> tuple[List[float], List[float]]:
        if not self.xs:
            return [], []
        pairs = sorted(zip(self.xs, self.ys), key=lambda pair: pair[0])
        xs, ys = zip(*pairs)
        return list(xs), list(ys)


def load_vis_records(path: Path) -> tuple[List[dict], List[dict]]:
    train_records: List[dict] = []
    val_records: List[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "iter" in record:
                train_records.append(record)
            else:
                val_records.append(record)
    return train_records, val_records


def build_metric_series(records: List[dict], x_key: str) -> Dict[str, MetricSeries]:
    metrics: Dict[str, MetricSeries] = {}
    for rec in records:
        if x_key not in rec:
            continue
        x_val = float(rec[x_key])
        if not math.isfinite(x_val):
            continue
        for key, value in rec.items():
            if key == x_key or key in IGNORE_METRICS:
                continue
            if not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                continue
            series = metrics.setdefault(key, MetricSeries(key))
            series.add_point(x_val, float(value))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics stored in vis_data JSON-lines file.")
    parser.add_argument("vis_json", type=Path, help="Path to vis_data JSON lines file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Folder to store figures (defaults to <vis_dir>/plots_vis).",
    )
    parser.add_argument(
        "--train-metrics",
        nargs="+",
        help="Training metrics to plot. Defaults to every metric that starts with 'loss'.",
    )
    parser.add_argument(
        "--val-metrics",
        nargs="+",
        help="Validation metrics to plot. Defaults to all scalar entries.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI (default: 160).")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.6,
        help="EMA smoothing factor 0-1, higher = smoother (default: 0.6, TensorBoard style).",
    )
    parser.add_argument(
        "--start-iter",
        type=float,
        default=None,
        help="Only plot records with iter/step >= this value.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Hide raw data, show only smoothed curves.",
    )
    parser.add_argument("--show", action="store_true", help="Display figures instead of only saving PNG files.")
    return parser.parse_args()


def choose_metric_names(
    metrics: Mapping[str, MetricSeries],
    requested: Sequence[str] | None,
    *,
    default_prefixes: Iterable[str] | None = None,
) -> List[str]:
    def is_ignored(name: str) -> bool:
        return any(token in name for token in IGNORE_METRICS)

    available = set(metrics.keys())
    if not available:
        return []
    if requested:
        return [name for name in requested if name in available and not is_ignored(name)]
    if default_prefixes:
        return sorted(
            name
            for name in available
            if any(name.startswith(prefix) for prefix in default_prefixes) and not is_ignored(name)
        )
    return sorted(name for name in available if not is_ignored(name))


def ensure_output_dir(vis_path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        output_dir = vis_path.parent / "plots_vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def exponential_moving_average(ys: List[float], smoothing: float = 0.6) -> List[float]:
    """
    Apply exponential moving average smoothing (TensorBoard style).
    
    Args:
        ys: Raw data points
        smoothing: Smoothing factor (0 = no smoothing, 1 = max smoothing)
                   TensorBoard default is around 0.6
    
    Returns:
        Smoothed data points
    """
    if not ys:
        return []
    
    smoothed = []
    last = ys[0]
    for y in ys:
        # EMA formula: smoothed = smoothing * last + (1 - smoothing) * current
        smoothed_val = smoothing * last + (1 - smoothing) * y
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_metrics(
    plt_module,
    metrics: Mapping[str, MetricSeries],
    names: Sequence[str],
    *,
    title: str = None,
    xlabel: str,
    dpi: int,
    smoothing: float = 0.6,
    show_raw: bool = True,
):
    """
    Plot metrics with TensorBoard-style smoothing.
    
    Args:
        plt_module: matplotlib.pyplot module
        metrics: Dictionary of metric series
        names: Names of metrics to plot
        title: Plot title
        xlabel: X-axis label
        dpi: Figure DPI
        smoothing: EMA smoothing factor (0-1, higher = smoother)
        show_raw: Whether to show raw data with low opacity
    """
    fig, ax = plt_module.subplots(figsize=(10, 6), dpi=dpi)
    
    # Use a nice color cycle
    colors = plt_module.cm.tab10.colors
    
    for idx, name in enumerate(names):
        xs, ys = metrics[name].sorted_points()
        if not xs:
            continue
        
        color = colors[idx % len(colors)]
        
        if show_raw and smoothing > 0:
            # Plot raw data with low opacity (light color)
            ax.plot(xs, ys, color=color, alpha=0.25, linewidth=0.8)
            
            # Plot smoothed data with high opacity (solid color)
            ys_smoothed = exponential_moving_average(ys, smoothing)
            ax.plot(xs, ys_smoothed, color=color, alpha=1.0, linewidth=2.0, label=name)
        else:
            # No smoothing, just plot raw data
            ax.plot(xs, ys, color=color, alpha=1.0, linewidth=1.5, label=name)
    
    # ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Value", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, alpha=0.3)
    # ax.legend(fontsize=14)
    ax.set_xlim(left=0)
    ax.margins(x=0)
    fig.tight_layout()
    return fig, ax


def main() -> None:
    args = parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    vis_path = args.vis_json.expanduser().resolve()
    if not vis_path.is_file():
        raise FileNotFoundError(f"vis_data JSON file not found: {vis_path}")

    train_records, val_records = load_vis_records(vis_path)
    if args.start_iter is not None:
        train_records = [rec for rec in train_records if float(rec.get("iter", -1)) >= args.start_iter]
        val_records = [rec for rec in val_records if float(rec.get("step", -1)) >= args.start_iter]
    train_metrics = build_metric_series(train_records, x_key="iter")
    val_metrics = build_metric_series(val_records, x_key="step")

    output_dir = ensure_output_dir(vis_path, args.output_dir)
    if args.train_metrics:
        train_names = choose_metric_names(train_metrics, args.train_metrics)
    else:
        # By default, only plot the aggregated total loss curve.
        train_names = [name for name in DEFAULT_TRAIN_METRICS if name in train_metrics]
    val_names = choose_metric_names(val_metrics, args.val_metrics)

    if not train_names and not val_names:
        print("No metrics matched. Try specifying --train-metrics/--val-metrics explicitly.")
        return

    if train_names:
        fig, _ = plot_metrics(
            plt_module=plt,
            metrics=train_metrics,
            names=train_names,
            # title=f"Training metrics ({vis_path.name})",
            xlabel="Iteration",
            dpi=args.dpi,
            smoothing=args.smoothing,
            show_raw=not args.no_raw,
        )
        out_path = output_dir / f"{vis_path.stem}_train.png"
        fig.savefig(out_path)
        if not args.show:
            plt.close(fig)
        print(f"Saved training plot -> {out_path}")
    else:
        print("No training metrics to plot.")

    if val_names:
        fig, _ = plot_metrics(
            plt_module=plt,
            metrics=val_metrics,
            names=val_names,
            # title=f"Validation metrics ({vis_path.name})",
            xlabel="Iteration",
            dpi=args.dpi,
            smoothing=args.smoothing,
            show_raw=not args.no_raw,
        )
        out_path = output_dir / f"{vis_path.stem}_val.png"
        fig.savefig(out_path)
        if not args.show:
            plt.close(fig)
        print(f"Saved validation plot -> {out_path}")
    else:
        print("No validation metrics to plot.")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
