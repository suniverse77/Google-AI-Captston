"""Heatmap overlay 시각화 유틸."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3:
            image = image.permute(1, 2, 0)
        image = image.numpy()
    elif hasattr(image, "detach") and hasattr(image, "cpu"):
        image = image.detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
    elif hasattr(image, "permute"):
        image = image.permute(1, 2, 0)

    if isinstance(image, np.ndarray):
        arr = image
    else:
        arr = np.array(image)

    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.max() > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def create_overlay(image: np.ndarray, heatmap: np.ndarray, *, alpha: float = 0.5, cmap: str = "plasma") -> np.ndarray:
    """원본 이미지와 heatmap을 겹쳐서 반환."""
    heatmap_norm = heatmap.astype(np.float32)
    if heatmap_norm.max() > 0:
        heatmap_norm /= heatmap_norm.max()
    colored = plt.get_cmap(cmap)(heatmap_norm)[..., :3]
    overlay = (1 - alpha) * image + alpha * colored
    return np.clip(overlay, 0.0, 1.0)


def plot_overlay_grid(
    *,
    image: np.ndarray,
    heatmap: np.ndarray,
    mask: Optional[np.ndarray] = None,
    overlay: Optional[np.ndarray] = None,
    titles: Optional[Iterable[str]] = None,
    figsize=(12, 4),
) -> plt.Figure:
    """원본, 마스크, heatmap, overlay를 4열로 시각화."""
    cols = ["Image", "Mask", "Heatmap", "Overlay"]
    if titles:
        cols = list(titles)

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    axes[0].imshow(image)
    axes[0].set_title(cols[0])

    axes[1].imshow(mask if mask is not None else np.zeros_like(heatmap), cmap="gray")
    axes[1].set_title(cols[1])

    axes[2].imshow(heatmap, cmap="plasma")
    axes[2].set_title(cols[2])

    overlay_img = overlay if overlay is not None else create_overlay(image, heatmap)
    axes[3].imshow(overlay_img)
    axes[3].set_title(cols[3])

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    return fig


def save_overlay_examples(
    samples: Iterable[Dict[str, np.ndarray]],
    *,
    output_dir: Path,
    cmap: str = "plasma",
    alpha: float = 0.5,
) -> None:
    """샘플 리스트를 받아 overlay 이미지를 저장."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples):
        image = _to_numpy(sample["image"])
        heatmap = _normalize_spatial(sample.get("heatmap"), image.shape[:2], mode="cubic")
        if heatmap is None:
            heatmap = np.zeros(image.shape[:2], dtype=np.float32)

        mask = _normalize_spatial(sample.get("mask"), image.shape[:2], mode="nearest")

        overlay = create_overlay(image, heatmap, alpha=alpha, cmap=cmap)

        fig = plot_overlay_grid(
            image=image,
            heatmap=heatmap,
            mask=mask,
            overlay=overlay,
            titles=sample.get("titles"),
        )
        save_path = output_dir / f"overlay_{idx:03d}.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)


__all__ = ["create_overlay", "plot_overlay_grid", "save_overlay_examples"]


def _normalize_spatial(
    array: Optional[np.ndarray],
    target_hw: Iterable[int],
    *,
    mode: str = "cubic",
) -> Optional[np.ndarray]:
    """Convert array to (H, W) numpy array and resize to target_hw."""
    if array is None:
        return None

    arr = _to_numpy(array)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        return None

    target_h, target_w = target_hw
    if arr.shape != (target_h, target_w):
        if target_h == 0 or target_w == 0:
            return None
        interpolation = cv2.INTER_CUBIC if mode == "cubic" else cv2.INTER_NEAREST
        arr = cv2.resize(arr, (target_w, target_h), interpolation=interpolation)

    return arr.astype(np.float32)

