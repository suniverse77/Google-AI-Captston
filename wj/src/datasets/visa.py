"""VisA dataset utilities aligned with the project BaseDataset API."""

from __future__ import annotations

import csv
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .base import BaseDataset

visa_classes = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "visa"
VISA_DIR = os.environ.get("VISA_DIR", str(DEFAULT_ROOT))


def _resolve_split_csv(root: Path, split_type: str, split_csv: Optional[os.PathLike]) -> Path:
    if split_csv:
        return Path(split_csv)
    return root / "split_csv" / f"{split_type}.csv"


def _read_split_csv(
    *,
    root: Path,
    csv_path: Path,
    category: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("object") != category:
                continue

            split = row.get("split", "").strip().lower()
            label = row.get("label", "").strip().lower()
            image_rel = row.get("image", "").strip()
            mask_rel = row.get("mask", "").strip()
            anomaly_type = row.get("anomaly") or row.get("anomaly_type")

            if not image_rel:
                continue

            image_path = root / image_rel
            mask_path = root / mask_rel if mask_rel else None

            label_value = 0 if label == "normal" else 1
            if label_value == 0:
                defect_type = "good"
            else:
                defect_type = anomaly_type.strip() if anomaly_type else "anomaly"

            record = {
                "image": str(image_path),
                "mask": str(mask_path) if mask_path else 0,
                "label": label_value,
                "defect_type": defect_type,
            }

            if split == "train":
                train_records.append(record)
            elif split == "test":
                test_records.append(record)

    train_records.sort(key=lambda item: item["image"])
    test_records.sort(key=lambda item: item["image"])

    return train_records, test_records


def load_visa(
    category: str,
    *,
    root: Optional[os.PathLike] = None,
    split_type: str = "1cls",
    split_csv: Optional[os.PathLike] = None,
    k_shot: int = 0,
    experiment_idx: int = 0,
    use_full_train_if_k_shot_zero: bool = False,
) -> Tuple[Tuple[List[str], List[Any], List[int], List[str]], Tuple[List[str], List[Any], List[int], List[str]]]:
    if category not in visa_classes:
        raise ValueError(f"Unknown VisA category: {category}")
    if k_shot not in [0, 1, 5, 10]:
        raise ValueError("k_shot must be one of [0, 1, 5, 10]")
    if experiment_idx not in [0, 1, 2]:
        raise ValueError("experiment_idx must be 0, 1, or 2")

    base_root = Path(root or VISA_DIR)
    csv_path = _resolve_split_csv(base_root, split_type, split_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"VisA split CSV가 존재하지 않습니다: {csv_path}")

    train_records, test_records = _read_split_csv(root=base_root, csv_path=csv_path, category=category)

    if k_shot == 0:
        if use_full_train_if_k_shot_zero:
            selected_train = list(train_records)
        else:
            selected_train = []
    else:
        rng = random.Random(f"{category}-{experiment_idx}-{k_shot}")
        if len(train_records) <= k_shot:
            selected_train = list(train_records)
        else:
            indices = sorted(rng.sample(range(len(train_records)), k_shot))
            selected_train = [train_records[idx] for idx in indices]

    def _unpack(records: List[Dict[str, Any]]) -> Tuple[List[str], List[Any], List[int], List[str]]:
        images = [entry["image"] for entry in records]
        masks = [entry["mask"] for entry in records]
        labels = [entry["label"] for entry in records]
        defect_types = [entry["defect_type"] for entry in records]
        return images, masks, labels, defect_types

    return _unpack(selected_train), _unpack(test_records)


class VisaDataset(BaseDataset):
    """Torch Dataset wrapping VisA loader output."""

    def __init__(
        self,
        *,
        category: str,
        split: str,
        root: Optional[os.PathLike] = None,
        split_type: str = "1cls",
        split_csv: Optional[os.PathLike] = None,
        k_shot: int = 0,
        experiment_idx: int = 0,
        use_full_train_if_k_shot_zero: bool = False,
        transforms=None,
        mask_transforms=None,
        include_masks: bool = True,
        max_samples: Optional[int] = None,
        test_file_numbers: Optional[List[int]] = None,
    ) -> None:
        super().__init__(transforms=transforms)
        self.category = category
        self.split = split
        self.split_type = split_type
        self.split_csv = split_csv
        self.k_shot = k_shot
        self.experiment_idx = experiment_idx
        self.use_full_train_if_k_shot_zero = use_full_train_if_k_shot_zero
        self.mask_transforms = mask_transforms
        self.include_masks = include_masks
        self.max_samples = max_samples
        self.test_file_numbers = test_file_numbers

        if root is not None:
            self.set_root(root)

        self.base_root = Path(VISA_DIR)
        if not self.base_root.exists():
            raise FileNotFoundError(f"VisA 데이터 루트를 찾을 수 없습니다: {self.base_root}")

        train_data, test_data = load_visa(
            category,
            root=self.base_root,
            split_type=self.split_type,
            split_csv=self.split_csv,
            k_shot=self.k_shot,
            experiment_idx=self.experiment_idx,
            use_full_train_if_k_shot_zero=self.use_full_train_if_k_shot_zero,
        )

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

        # 특정 파일 번호로 필터링 (test split에만 적용)
        if split == "test" and self.test_file_numbers is not None and len(self.test_file_numbers) > 0:
            filtered_indices = []
            for idx, img_path in enumerate(self.image_paths):
                filename = Path(img_path).stem
                match = re.search(r'(\d+)', filename)
                if match:
                    file_number = int(match.group(1))
                    if file_number in self.test_file_numbers:
                        filtered_indices.append(idx)
            
            if filtered_indices:
                self.image_paths = [self.image_paths[i] for i in filtered_indices]
                self.mask_paths = [self.mask_paths[i] for i in filtered_indices]
                self.labels = [self.labels[i] for i in filtered_indices]
                self.defect_types = [self.defect_types[i] for i in filtered_indices]

        if self.max_samples is not None and self.max_samples > 0 and len(self.image_paths) > self.max_samples:
            indices = self._select_indices(len(self.image_paths), self.max_samples)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.defect_types = [self.defect_types[i] for i in indices]

    @staticmethod
    def set_root(root: os.PathLike) -> None:
        global VISA_DIR
        resolved = Path(root).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"VisA 루트 경로가 존재하지 않습니다: {resolved}")
        VISA_DIR = str(resolved)
        os.environ["VISA_DIR"] = VISA_DIR

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


__all__ = ["visa_classes", "load_visa", "VisaDataset", "VISA_DIR"]

