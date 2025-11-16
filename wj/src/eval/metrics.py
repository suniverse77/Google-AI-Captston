"""정량 평가 메트릭 유틸리티.

WinCLIP 원본 `utils.metrics` 모듈과 연동해 중복 계산을 피하도록 통합했다.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn import metrics as sk_metrics

from .winclip_metrics import metric_cal

def _safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(sk_metrics.roc_auc_score(labels, scores))


def compute_image_level_auroc(image_scores: Sequence[float], labels: Sequence[int]) -> float:
    """이미지 단위 AUROC."""
    return _safe_auc(labels, image_scores)


def compute_pixel_level_auroc(pixel_scores: np.ndarray, pixel_labels: np.ndarray) -> float:
    """픽셀 단위 AUROC. 입력은 (N, H, W)."""
    scores = pixel_scores.reshape(pixel_scores.shape[0], -1)
    labels = pixel_labels.reshape(pixel_labels.shape[0], -1)
    return _safe_auc(labels.flatten().tolist(), scores.flatten().tolist())


def summarize_metrics(
    image_scores: Sequence[float],
    image_labels: Sequence[int],
    pixel_scores: Optional[np.ndarray] = None,
    pixel_labels: Optional[np.ndarray] = None,
    *,
    pro_cfg: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """메트릭 계산 결과를 하나의 dict로 정리."""
    summary: Dict[str, float] = {}
    summary["image_auroc"] = compute_image_level_auroc(image_scores, image_labels)

    if pixel_scores is not None and pixel_labels is not None:
        pixel_scores_np = np.asarray(pixel_scores)
        pixel_labels_np = np.asarray(pixel_labels)
        if pixel_scores_np.ndim == 4:
            pixel_scores_np = pixel_scores_np[:, 0]  # cursor 수정 1108
        if pixel_labels_np.ndim == 4:
            pixel_labels_np = pixel_labels_np[:, 0]  # cursor 수정 1108
        cal_pro = True
        if pro_cfg and "enabled" in pro_cfg:
            cal_pro = bool(pro_cfg["enabled"])

        winclip_metrics = metric_cal(
            pixel_scores_np,
            list(image_labels),
            pixel_labels_np,
            cal_pro=cal_pro,
        )

        summary.update(
            {
                "image_f1": winclip_metrics["i_f1"] / 100.0,
                "pixel_f1": winclip_metrics["p_f1"] / 100.0,
                "region_f1": winclip_metrics["r_f1"] / 100.0,
            }
        )

        summary["pixel_auroc"] = winclip_metrics["p_roc"] / 100.0

        if cal_pro:
            summary["pro_auroc"] = winclip_metrics["p_pro"] / 100.0

    return summary


def save_metrics_csv(metrics: Mapping[str, float], output_path: Path) -> None:
    """metrics dict를 CSV 한 줄로 저장."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


__all__ = [
    "compute_image_level_auroc",
    "compute_pixel_level_auroc",
    "summarize_metrics",
    "save_metrics_csv",
]

