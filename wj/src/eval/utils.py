"""Copied from WinCLIP (wj/src/WinClip/utils/eval_utils.py)."""

import os
from typing import Iterable, Tuple

import cv2
import numpy as np


def specify_resolution(
    image_list: Iterable[np.ndarray],
    score_list: Iterable[np.ndarray],
    mask_list: Iterable[np.ndarray],
    resolution: Tuple[int, int] = (400, 400),
):
    resize_image = []
    resize_score = []
    resize_mask = []
    for image, score, mask in zip(image_list, score_list, mask_list):
        image = cv2.resize(image, (resolution[0], resolution[1]), interpolation=cv2.INTER_CUBIC)
        score = cv2.resize(score, (resolution[0], resolution[1]), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (resolution[0], resolution[1]), interpolation=cv2.INTER_NEAREST)
        resize_image += [image]
        resize_score += [score]
        resize_mask += [mask]

    return resize_image, resize_score, resize_mask


def normalize(scores: np.ndarray) -> np.ndarray:
    max_value = np.max(scores)
    min_value = np.min(scores)
    norml_scores = (scores - min_value) / (max_value - min_value)
    return norml_scores


def save_single_result(
    classification_score,
    segmentation_score,
    root_dir,
    shot_name,
    experiment_indx,
    subset_name,
    defect_type,
    name,
    use_defect_type,
):
    if use_defect_type:
        save_dir = os.path.join(root_dir, shot_name, experiment_indx, subset_name, defect_type)
    else:
        save_dir = os.path.join(root_dir, shot_name, experiment_indx, subset_name)

    os.makedirs(save_dir, exist_ok=True)

    classification_dir = os.path.join(save_dir, 'classification')
    segmentation_dir = os.path.join(save_dir, 'segmentation')
    os.makedirs(classification_dir, exist_ok=True)
    os.makedirs(segmentation_dir, exist_ok=True)

    classification_path = os.path.join(classification_dir, f'{name}.txt')
    segmentation_path = os.path.join(segmentation_dir, f'{name}.npz')

    with open(classification_path, "w") as f:
        f.write(f'{classification_score:.5f}')

    segmentation_score = np.round(segmentation_score * 255).astype(np.uint8)
    np.savez_compressed(segmentation_path, img=segmentation_score)


def save_results(
    classification_score_list,
    segmentation_score_list,
    root_dir,
    shot_name,
    experiment_indx,
    name_list,
    use_defect_type,
):
    for classification_score, segmentation_score, full_name in zip(
        classification_score_list,
        segmentation_score_list,
        name_list,
    ):
        subset_name, defect_type, name = full_name.split('-')
        save_single_result(
            classification_score,
            segmentation_score,
            root_dir,
            shot_name,
            experiment_indx,
            subset_name,
            defect_type,
            name,
            use_defect_type,
        )


__all__ = [
    "specify_resolution",
    "normalize",
    "save_single_result",
    "save_results",
]

