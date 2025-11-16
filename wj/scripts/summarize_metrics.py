#!/usr/bin/env python3
"""Summarize WinCLIP/PatchCore evaluation metrics.

Given a CSV in the same format as `summary_metrics.csv`, this script
produces a tidy table that focuses on the key metrics per level:
  - image-level: AUROC, AP, max-F1
  - pixel-level: AUROC, AP, max-F1

If the requested AP columns are missing in the input CSV, the values
are filled with NaN and a warning is emitted so you can decide whether
to supply alternative columns (e.g. PRO) via command-line flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _resolve_column(
    df: pd.DataFrame,
    desired: str,
    fallback: Optional[str],
    level: str,
    metric: str,
) -> Optional[str]:
    if desired and desired in df.columns:
        return desired
    if fallback and fallback in df.columns:
        return fallback
    print(
        f"[WARN] Column for {level} {metric} not found "
        f"(tried '{desired}'"
        f"{f' and fallback {fallback!r}' if fallback else ''}). Filling with NaN.",
        file=sys.stderr,
    )
    return None


def _build_mapping(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Dict[str, Optional[str]]]:
    image_ap_col = _resolve_column(
        df,
        desired=args.image_ap_column,
        fallback=None,
        level="image",
        metric="AP",
    )
    pixel_ap_col = _resolve_column(
        df,
        desired=args.pixel_ap_column,
        fallback=args.pixel_ap_fallback,
        level="pixel",
        metric="AP",
    )
    mapping = {
        "image": {
            "auroc": "image_auroc" if "image_auroc" in df.columns else None,
            "ap": image_ap_col,
            "max_f1": "image_f1" if "image_f1" in df.columns else None,
        },
        "pixel": {
            "auroc": "pixel_auroc" if "pixel_auroc" in df.columns else None,
            "ap": pixel_ap_col,
            "max_f1": "pixel_f1" if "pixel_f1" in df.columns else None,
        },
    }
    for level, metrics in mapping.items():
        for metric, col in metrics.items():
            if col is None:
                print(
                    f"[WARN] Column for {level} {metric} missing. Values will be NaN.",
                    file=sys.stderr,
                )
    return mapping


def summarize_metrics(df: pd.DataFrame, mapping: Dict[str, Dict[str, Optional[str]]]) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        for level, metrics in mapping.items():
            record = {"category": row["category"], "level": level}
            for metric, column_name in metrics.items():
                value = np.nan
                if column_name is not None:
                    value = row.get(column_name, np.nan)
                    # Remove units if the CSV stored formatted strings (e.g., "64.100 ms").
                    if isinstance(value, str):
                        try:
                            value = float(value.split()[0].replace(",", ""))
                        except ValueError:
                            value = np.nan
                record[metric] = value
            records.append(record)
    result_df = pd.DataFrame.from_records(records)
    return result_df


def add_macro_average(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("level", as_index=False)[["auroc", "ap", "max_f1"]].mean()
    grouped.insert(0, "category", "mean")
    return pd.concat([df, grouped], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize evaluation metrics CSV.")
    parser.add_argument("input", type=Path, help="Path to summary_metrics.csv")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where to write the summarized CSV. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--include-mean",
        action="store_true",
        help="Append macro-average rows per level.",
    )
    parser.add_argument(
        "--image-ap-column",
        default="image_ap",
        help="Column name to use for image-level AP (default: image_ap).",
    )
    parser.add_argument(
        "--pixel-ap-column",
        default="pixel_ap",
        help="Preferred column name for pixel-level AP (default: pixel_ap).",
    )
    parser.add_argument(
        "--pixel-ap-fallback",
        default=None,
        help="Fallback column for pixel-level AP if the preferred column is missing (e.g., pro_auroc).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    mapping = _build_mapping(df, args)
    summary_df = summarize_metrics(df, mapping)
    if args.include_mean:
        summary_df = add_macro_average(summary_df)

    summary_df = summary_df.sort_values(["category", "level"]).reset_index(drop=True)
    numeric_cols = ["auroc", "ap", "max_f1"]
    summary_df[numeric_cols] = summary_df[numeric_cols].apply(lambda col: (col * 100).round(2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.output, index=False)
        print(f"[INFO] Wrote summary to {args.output}")
    else:
        print(summary_df.to_csv(index=False), end="")


if __name__ == "__main__":
    main()

