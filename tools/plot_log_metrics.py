#!/usr/bin/env python3
"""Plot losses and validation metrics from MMEngine/MMPose logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


EPOCH_LINE_RE = re.compile(
    r"Epoch\((?P<phase>train|val)\)\s+\[(?P<epoch>\d+)\]\s*\[(?P<cur>\d+)\s*/\s*(?P<total>\d+)\]"
)
ITER_LINE_RE = re.compile(
    r"Iter\((?P<phase>train|val)\)\s*\[\s*(?P<cur>\d+)\s*/\s*(?P<total>\d+)\]"
)
VALUE_RE = re.compile(
    r"([A-Za-z0-9_@./-]+):\s*([-+]?inf|[-+]?nan|[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)
DEFAULT_TRAIN_PREFIXES = ("loss",)
IGNORE_METRICS = {"time", "data_time", "memory", "eta", "lr"}


@dataclass
class MetricSeries:
    """Container storing a single metric evolution."""

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


class LogParser:
    """Parse mmengine formatted logs and gather train/val metric curves."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._val_epoch_counter = 0

    def parse(self) -> tuple[Dict[str, MetricSeries], Dict[str, MetricSeries]]:
        train: MutableMapping[str, MetricSeries] = {}
        val: MutableMapping[str, MetricSeries] = {}

        with self.log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                epoch_info = EPOCH_LINE_RE.search(line)
                iter_info = None if epoch_info else ITER_LINE_RE.search(line)
                if not epoch_info and not iter_info:
                    continue

                scalars = self._extract_scalars(line)
                if not scalars:
                    continue

                if epoch_info:
                    phase = epoch_info.group("phase")
                    epoch = int(epoch_info.group("epoch"))
                    cur_iter = int(epoch_info.group("cur"))
                    total_iter = int(epoch_info.group("total"))

                    if phase == "train":
                        global_iter = (epoch - 1) * total_iter + cur_iter
                        self._add_metrics(train, scalars, x_value=global_iter)
                    else:
                        if cur_iter != total_iter:
                            continue
                        self._add_metrics(val, scalars, x_value=epoch)
                elif iter_info:
                    phase = iter_info.group("phase")
                    cur_iter = int(iter_info.group("cur"))
                    total_iter = int(iter_info.group("total"))

                    if phase == "train":
                        self._add_metrics(train, scalars, x_value=cur_iter)
                    else:
                        if cur_iter != total_iter:
                            continue
                        self._val_epoch_counter += 1
                        self._add_metrics(val, scalars, x_value=self._val_epoch_counter)

        return dict(train), dict(val)

    @staticmethod
    def _extract_scalars(line: str) -> Dict[str, float]:
        scalars: Dict[str, float] = {}
        for key, value in VALUE_RE.findall(line):
            if not any(char.isalpha() for char in key):
                continue
            try:
                scalars[key] = float(value)
            except ValueError:
                continue
        return scalars

    @staticmethod
    def _add_metrics(
        storage: MutableMapping[str, MetricSeries], scalars: Mapping[str, float], *, x_value: float
    ) -> None:
        for name, y_value in scalars.items():
            series = storage.setdefault(name, MetricSeries(name))
            series.add_point(x_value, y_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MMEngine/MMPose losses and validation metrics.")
    parser.add_argument("log", type=Path, help="Path to the training log file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Folder to store figures (defaults to <log_dir>/plots).",
    )
    parser.add_argument(
        "--train-metrics",
        nargs="+",
        help="Training metrics to plot. Defaults to every metric that starts with 'loss'.",
    )
    parser.add_argument(
        "--val-metrics",
        nargs="+",
        help="Validation metrics to plot. Defaults to every scalar reported when validation finishes.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI (default: 160).")
    parser.add_argument(
        "--show", action="store_true", help="Display figures instead of only saving PNG files."
    )
    return parser.parse_args()


def choose_metric_names(
    metrics: Mapping[str, MetricSeries],
    requested: Sequence[str] | None,
    *,
    default_prefixes: Iterable[str] | None = None,
) -> List[str]:
    available = set(metrics.keys())
    if not available:
        return []

    if requested:
        return [name for name in requested if name in available]

    if default_prefixes:
        return sorted(
            name
            for name in available
            if any(name.startswith(prefix) for prefix in default_prefixes) and name not in IGNORE_METRICS
        )

    return sorted(name for name in available if name not in IGNORE_METRICS)


def ensure_output_dir(log_path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        output_dir = log_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_metrics(
    plt_module,
    metrics: Mapping[str, MetricSeries],
    names: Sequence[str],
    *,
    title: str,
    xlabel: str,
    dpi: int,
) -> tuple[object, Path]:
    fig, ax = plt_module.subplots(figsize=(10, 6), dpi=dpi)
    for name in names:
        xs, ys = metrics[name].sorted_points()
        if not xs:
            continue
        ax.plot(xs, ys, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def main() -> None:
    args = parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    log_path = args.log.expanduser().resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    parser = LogParser(log_path)
    train_metrics, val_metrics = parser.parse()
    output_dir = ensure_output_dir(log_path, args.output_dir)

    train_names = choose_metric_names(train_metrics, args.train_metrics, default_prefixes=DEFAULT_TRAIN_PREFIXES)
    val_names = choose_metric_names(val_metrics, args.val_metrics)

    if not train_names and not val_names:
        print("No metrics matched. Try specifying --train-metrics/--val-metrics explicitly.")
        return

    if train_names:
        fig, _ = plot_metrics(
            plt_module=plt,
            metrics=train_metrics,
            names=train_names,
            title=f"Training metrics ({log_path.name})",
            xlabel="Iteration",
            dpi=args.dpi,
        )
        out_path = output_dir / f"{log_path.stem}_train.png"
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
            title=f"Validation metrics ({log_path.name})",
            xlabel="Epoch",
            dpi=args.dpi,
        )
        out_path = output_dir / f"{log_path.stem}_val.png"
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
