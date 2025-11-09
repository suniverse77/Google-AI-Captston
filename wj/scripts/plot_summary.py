"""summary.csv 기반 자동 시각화."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot WinCLIP summary metrics.")
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="summary.csv 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="플롯 저장 디렉터리 (기본: summary.csv와 동일한 디렉터리/plots)",
    )
    return parser.parse_args()


def load_summary(summary_path: Path) -> List[Dict[str, float]]:
    with summary_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = []
        for row in reader:
            rows.append({k: (float(v) if k != "category" else v) for k, v in row.items()})
    return rows


def plot_bar_metric(rows: List[Dict[str, float]], metric: str, output_path: Path) -> None:
    categories = [row["category"] for row in rows]
    values = [row.get(metric, np.nan) for row in rows]
    plt.figure(figsize=(10, 4))
    plt.bar(categories, values)
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{metric} per category")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_scatter(rows: List[Dict[str, float]], x_metric: str, y_metric: str, output_path: Path) -> None:
    xs = [row.get(x_metric, np.nan) for row in rows]
    ys = [row.get(y_metric, np.nan) for row in rows]
    labels = [row["category"] for row in rows]
    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y))
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(f"{y_metric} vs {x_metric}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    rows = load_summary(summary_path)
    if not rows:
        raise RuntimeError("summary.csv가 비어 있습니다.")

    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("image_auroc", "pixel_auroc", "aupro"):
        if metric in rows[0]:
            plot_bar_metric(rows, metric, output_dir / f"{metric}_bar.png")

    if "fps" in rows[0]:
        plot_scatter(rows, "fps", "image_auroc", output_dir / "fps_vs_image_auroc.png")

    # TODO: ROC / PRO 곡선 플롯을 위해서는 per-threshold 결과를 저장/불러오는 로직이 필요합니다.

    print(f"[INFO] Plots saved to {output_dir}")


if __name__ == "__main__":
    main()

