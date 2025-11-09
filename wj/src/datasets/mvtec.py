"""MVTec-AD dataset utilities (WinCLIP compatibility + PyTorch dataset)."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .base import BaseDataset

mvtec_classes = [
    'carpet',
    'grid',
    'leather',
    'tile',
    'wood',
    'bottle',
    'cable',
    'capsule',
    'hazelnut',
    'metal_nut',
    'pill',
    'screw',
    'toothbrush',
    'transistor',
    'zipper',
]

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "mvtec"
MVTEC2D_DIR = os.environ.get('MVTEC2D_DIR', str(DEFAULT_ROOT))


def load_mvtec(category, k_shot, experiment_indx):
    def load_phase(root_path, gt_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(root_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.png")
                gt_paths = [
                    os.path.join(gt_path, defect_type, os.path.basename(s)[:-4] + '_mask.png') for s in img_paths
                ]
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in mvtec_classes
    assert k_shot in [0, 1, 5, 10]
    assert experiment_indx in [0, 1, 2]

    test_img_path = os.path.join(MVTEC2D_DIR, category, 'test')
    train_img_path = os.path.join(MVTEC2D_DIR, category, 'train')
    ground_truth_path = os.path.join(MVTEC2D_DIR, category, 'ground_truth')

    if k_shot == 0:
        training_indx = []
    else:
        seed_file = os.path.join(os.path.dirname(__file__), 'seeds_mvtec', category, 'selected_samples_per_run.txt')
        with open(seed_file, 'r') as f:
            files = f.readlines()
        begin_str = f'{experiment_indx}-{k_shot}: '

        training_indx = []
        for line in files:
            if line.count(begin_str) > 0:
                strip_line = line[len(begin_str):-1]
                index = strip_line.split(' ')
                training_indx = index

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types = load_phase(
        train_img_path, ground_truth_path
    )

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types = load_phase(
        test_img_path, ground_truth_path
    )

    selected_train_img_tot_paths = []
    selected_train_gt_tot_paths = []
    selected_train_tot_labels = []
    selected_train_tot_types = []

    for img_path, gt_path, label, defect_type in zip(
        train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types
    ):
        if os.path.basename(img_path[:-4]) in training_indx:
            selected_train_img_tot_paths.append(img_path)
            selected_train_gt_tot_paths.append(gt_path)
            selected_train_tot_labels.append(label)
            selected_train_tot_types.append(defect_type)

    return (
        (
            selected_train_img_tot_paths,
            selected_train_gt_tot_paths,
            selected_train_tot_labels,
            selected_train_tot_types,
        ),
        (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types),
    )


class MvtecDataset(BaseDataset):
    """Torch Dataset wrapping WinCLIP's MVTec loader output."""

    def __init__(
        self,
        *,
        category: str,
        split: str,
        root: Optional[os.PathLike] = None,
        k_shot: int = 0,
        experiment_idx: int = 0,
        transforms=None,
        mask_transforms=None,
        include_masks: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__(transforms=transforms)
        self.category = category
        self.split = split
        self.k_shot = k_shot
        self.experiment_idx = experiment_idx
        self.mask_transforms = mask_transforms
        self.include_masks = include_masks
        self.max_samples = max_samples

        if root is not None:
            self.set_root(root)

        (train_data, test_data) = load_mvtec(category, k_shot, experiment_idx)
        if split == "train":
            self.image_paths, self.mask_paths, self.labels, self.defect_types = train_data
        elif split == "test":
            self.image_paths, self.mask_paths, self.labels, self.defect_types = test_data
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.image_paths = list(self.image_paths)
        self.mask_paths = list(self.mask_paths)
        self.labels = list(self.labels)
        self.defect_types = list(self.defect_types)

        if self.max_samples is not None and self.max_samples > 0 and len(self.image_paths) > self.max_samples:
            indices = self._select_indices(len(self.image_paths), self.max_samples)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.defect_types = [self.defect_types[i] for i in indices]

    @staticmethod
    def set_root(root: os.PathLike) -> None:
        global MVTEC2D_DIR
        MVTEC2D_DIR = str(root)
        os.environ["MVTEC2D_DIR"] = MVTEC2D_DIR

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        label = int(self.labels[index])
        defect_type = self.defect_types[index]

        image = Image.open(image_path).convert("RGB")
        image_tensor = self._to_tensor(image)

        mask_tensor = torch.zeros(1, image_tensor.shape[1], image_tensor.shape[2], dtype=torch.float32)
        if self.include_masks and mask_path and mask_path != 0:
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self._to_mask_tensor(mask, image.size)

        if self.transforms:
            image_tensor = self.transforms(image_tensor)
        if self.mask_transforms:
            mask_tensor = self.mask_transforms(mask_tensor)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "meta": {
                "path": image_path,
                "category": self.category,
                "defect_type": defect_type,
            },
        }

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor

    @staticmethod
    def _to_mask_tensor(mask: Image.Image, image_size: Tuple[int, int]) -> torch.Tensor:
        mask = mask.resize(image_size, Image.NEAREST)
        mask_array = np.array(mask, dtype=np.float32)
        mask_array = (mask_array > 0).astype(np.float32)
        return torch.from_numpy(mask_array).unsqueeze(0)

    @staticmethod
    def _select_indices(total: int, limit: int) -> List[int]:
        if limit >= total:
            return list(range(total))
        linspace = np.linspace(0, total - 1, limit, dtype=int)
        return sorted(set(linspace.tolist()))


__all__ = ["mvtec_classes", "load_mvtec", "MVTEC2D_DIR", "MvtecDataset"]

