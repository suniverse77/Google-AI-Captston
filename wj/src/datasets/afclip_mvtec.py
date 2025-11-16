"""AF-CLIP dataset adapter for the project pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .base import BaseDataset, build_dataloader

AF_CLIP_ROOT = Path(__file__).resolve().parents[2] / "git_clone" / "AF-CLIP"
if not AF_CLIP_ROOT.exists():
    raise ImportError(
        "AF-CLIP 서브모듈이 존재하지 않습니다. "
        "git_clone/AF-CLIP 경로가 올바른지 확인하세요."
    )

AF_CLIP_PATH_STR = str(AF_CLIP_ROOT)
if AF_CLIP_PATH_STR not in sys.path:
    sys.path.insert(0, AF_CLIP_PATH_STR)

try:
    from dataset.mvtec import MVTecDataset as AFClipRawMvtecDataset  # type: ignore
except ImportError as exc:  # pragma: no cover - 환경 의존
    raise ImportError(
        "AF-CLIP의 MVTecDataset 을 import 할 수 없습니다. "
        "필요한 의존성이 설치되어 있는지 확인하세요."
    ) from exc


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


class AFClipMvtecDataset(BaseDataset):
    """Wrap AF-CLIP's MVTec dataset to return pipeline-friendly samples."""

    def __init__(
        self,
        *,
        category: Optional[str],
        split: str,
        root: Optional[Path],
        image_transform: transforms.Compose,
        mask_transform: transforms.Compose,
        fewshot: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError(f"지원하지 않는 split 입니다: {split}")

        data_root = Path(root) if root is not None else _default_root()
        dataset_root = data_root / "mvtec"
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"MVTec 데이터셋 경로를 찾을 수 없습니다: {dataset_root}"
            )

        rng_state = np.random.get_state()
        if seed is not None:
            np.random.seed(seed)
        try:
            self._dataset: Dataset = AFClipRawMvtecDataset(
                root=str(data_root),
                train=split == "train",
                category=category,
                fewshot=fewshot if split == "train" else 0,
                transform=image_transform,
                gt_target_transform=mask_transform,
            )
        finally:
            np.random.set_state(rng_state)

        self.category = category
        self.split = split
        self.dataset_name = getattr(self._dataset, "dataset_name", "mvtec")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image, label, mask, category, image_path = self._dataset[index]

        if not torch.is_tensor(image):
            raise TypeError("AF-CLIP 데이터셋은 tensor 이미지를 반환해야 합니다.")
        image_tensor = image.to(torch.float32)

        if torch.is_tensor(mask):
            mask_tensor = mask.to(torch.float32)
        else:
            raise TypeError("AF-CLIP 데이터셋 마스크는 tensor 이어야 합니다.")

        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.dim() > 3:
            raise ValueError("마스크 텐서 차원이 올바르지 않습니다.")

        label_tensor = torch.tensor(int(label), dtype=torch.long)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": label_tensor,
            "meta": {
                "category": category,
                "path": image_path,
            },
        }


def _build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def _build_mask_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
    )


def build_afclip_mvtec_dataloaders(
    config: MutableMapping[str, Any],
    *,
    category: str,
    transforms=None,
    mask_transforms=None,
) -> Dict[str, Any]:
    loader_cfg = config.get("loader", {})
    batch_size = int(config.get("batch_size", 4))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", False))

    image_size = int(config.get("image_size", 518))
    train_fewshot = int(loader_cfg.get("fewshot", 0))
    seed = loader_cfg.get("seed")
    train_seed = loader_cfg.get("train_seed", seed)
    test_seed = loader_cfg.get("test_seed", seed)

    image_transform = transforms or _build_image_transform(image_size)
    mask_transform = mask_transforms or _build_mask_transform(image_size)

    root = config.get("root")
    train_dataset = AFClipMvtecDataset(
        category=category,
        split="train",
        root=Path(root) if root is not None else None,
        image_transform=image_transform,
        mask_transform=mask_transform,
        fewshot=train_fewshot,
        seed=train_seed,
    )
    test_dataset = AFClipMvtecDataset(
        category=category,
        split="test",
        root=Path(root) if root is not None else None,
        image_transform=image_transform,
        mask_transform=mask_transform,
        fewshot=0,
        seed=test_seed,
    )

    loader_cfg_dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": False,
    }

    loaders = {
        "train": build_dataloader(train_dataset, loader_cfg_dict, shuffle_override=False),
        "test": build_dataloader(test_dataset, loader_cfg_dict, shuffle_override=False),
    }

    return {
        "datasets": {"train": train_dataset, "test": test_dataset},
        "loaders": loaders,
    }


__all__ = [
    "AFClipMvtecDataset",
    "build_afclip_mvtec_dataloaders",
]

