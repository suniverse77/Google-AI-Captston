"""MVTec-AD 카테고리별 metrics.csv를 취합해 summary.csv 생성."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect WinCLIP evaluation metrics.")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="카테고리별 metrics.csv가 위치한 디렉터리 (예: results/mvtec/winclip/20240101-123000).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="summary.csv 저장 경로 (기본: <results-dir>/summary.csv).",
    )
    return parser.parse_args()


def read_metrics_file(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        header = next(reader)
        if header != ["metric", "value"]:
            raise ValueError(f"Unexpected header in {path}")
        metrics = {row[0]: float(row[1]) for row in reader}
    return metrics


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(results_dir)

    rows: List[Dict[str, float]] = []
    for category_dir in results_dir.iterdir():
        if not category_dir.is_dir():
            continue
        metrics_path = category_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = read_metrics_file(metrics_path)
        metrics["category"] = category_dir.name
        rows.append(metrics)

    if not rows:
        raise RuntimeError("metrics.csv 파일을 찾지 못했습니다.")

    output_path = Path(args.output) if args.output else results_dir / "summary.csv"
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Wrote summary to {output_path}")


if __name__ == "__main__":
    main()

