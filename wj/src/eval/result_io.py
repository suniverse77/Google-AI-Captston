"""Copied from WinCLIP (wj/src/WinClip/utils/csv_utils.py)."""

import os
from typing import Dict, Iterable, List

import pandas as pd


def write_results(results: Dict[str, float], cur_class: str, total_classes: Iterable[str], csv_path: str) -> None:
    keys = list(results.keys())

    if not os.path.exists(csv_path):
        df_all = None
        for class_name in total_classes:
            row = {k: 0.00 for k in keys}
            df_temp = pd.DataFrame(row, index=[class_name])
            if df_all is None:
                df_all = df_temp
            else:
                df_all = pd.concat([df_all, df_temp], axis=0)

        df_all.to_csv(csv_path, header=True, float_format='%.2f')

    df = pd.read_csv(csv_path, index_col=0)
    for k in keys:
        df.loc[cur_class, k] = results[k]

    df.to_csv(csv_path, header=True, float_format='%.2f')


def save_metric(metrics: Dict[str, float], total_classes: List[str], class_name: str, dataset: str, csv_path: str) -> None:
    total_classes = list(total_classes)
    if dataset != 'mvtec':
        for idx in range(len(total_classes)):
            total_classes[idx] = f"{dataset}-{total_classes[idx]}"
        class_name = f"{dataset}-{class_name}"
    write_results(metrics, class_name, total_classes, csv_path)


__all__ = ["write_results", "save_metric"]

