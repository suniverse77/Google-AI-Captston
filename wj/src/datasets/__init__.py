"""Dataset helper functions."""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from torch.utils.data import DataLoader

from .base import build_dataloader
from .mvtec import MvtecDataset
from .afclip_mvtec import AFClipMvtecDataset, build_afclip_mvtec_dataloaders
from .visa import VisaDataset

__all__ = [
    "build_mvtec_dataloaders",
    "create_mvtec_dataset",
    "build_visa_dataloaders",
    "create_visa_dataset",
    "build_afclip_mvtec_dataloaders",
    "AFClipMvtecDataset",
]


def _sanitize_limit(value):
    if value is None:
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    return value_int if value_int > 0 else None


def _build_image_transforms(config: MutableMapping[str, Any]) -> Optional[transforms.Compose]:
    tf_cfg = config.get("transforms", {})
    pipeline = []

    resize = tf_cfg.get("resize")
    if resize:
        pipeline.append(transforms.Resize(int(resize), interpolation=InterpolationMode.BILINEAR))

    center_crop = tf_cfg.get("center_crop")
    if center_crop:
        pipeline.append(transforms.CenterCrop(int(center_crop)))

    norm_cfg = tf_cfg.get("normalization")
    if norm_cfg:
        mean = norm_cfg.get("mean")
        std = norm_cfg.get("std")
        if mean is None or std is None:
            raise ValueError("normalization 설정에는 mean과 std가 모두 필요합니다.")
        pipeline.append(transforms.Normalize(mean=mean, std=std))

    if pipeline:
        return transforms.Compose(pipeline)
    return None


def _build_mask_transforms(config: MutableMapping[str, Any]) -> Optional[transforms.Compose]:
    tf_cfg = config.get("transforms", {})
    pipeline = []

    resize = tf_cfg.get("resize")
    if resize:
        pipeline.append(
            transforms.Resize(
                int(resize),
                interpolation=InterpolationMode.NEAREST,
                antialias=False,
            )
        )

    center_crop = tf_cfg.get("center_crop")
    if center_crop:
        pipeline.append(transforms.CenterCrop(int(center_crop)))

    if pipeline:
        return transforms.Compose(pipeline)
    return None


def build_mvtec_dataloaders(
    config: MutableMapping[str, Any],
    *,
    category: str,
    transforms=None,
    mask_transforms=None,
) -> Dict[str, Any]:
    loader_cfg = config.get("loader", {})
    k_shot = loader_cfg.get("k_shot", 0)
    experiment_idx = loader_cfg.get("experiment_idx", 0)
    root = config.get("root")
    use_full_train = bool(loader_cfg.get("use_full_train", False))

    default_limit = _sanitize_limit(loader_cfg.get("max_samples"))
    train_limit = _sanitize_limit(loader_cfg.get("max_train_samples")) or default_limit
    test_limit = _sanitize_limit(loader_cfg.get("max_test_samples")) or default_limit
    test_file_numbers = loader_cfg.get("test_file_numbers")  # 특정 파일 번호 리스트
    test_defect_types = loader_cfg.get("test_defect_types")  # 특정 defect_type 리스트

    image_transforms = transforms or _build_image_transforms(config)
    mask_transforms = mask_transforms or _build_mask_transforms(config)

    train_dataset = MvtecDataset(
        category=category,
        split="train",
        root=root,
        k_shot=k_shot,
        experiment_idx=experiment_idx,
        transforms=image_transforms,
        mask_transforms=mask_transforms,
        max_samples=train_limit,
        use_full_train_if_k_shot_zero=use_full_train,
    )
    test_dataset = MvtecDataset(
        category=category,
        split="test",
        root=root,
        k_shot=k_shot,
        experiment_idx=experiment_idx,
        transforms=image_transforms,
        mask_transforms=mask_transforms,
        max_samples=test_limit,
        use_full_train_if_k_shot_zero=use_full_train,
        test_file_numbers=test_file_numbers,
        test_defect_types=test_defect_types,
    )

    dataloader_cfg = {
        "batch_size": config.get("batch_size", 1),
        "num_workers": config.get("num_workers", 0),
        "pin_memory": config.get("pin_memory", False),
    }

    loaders = {
        "train": build_dataloader(train_dataset, dataloader_cfg, shuffle_override=False),
        "test": build_dataloader(test_dataset, dataloader_cfg, shuffle_override=False),
    }

    return {"datasets": {"train": train_dataset, "test": test_dataset}, "loaders": loaders}


def create_mvtec_dataset(
    config: MutableMapping[str, Any],
    *,
    category: str,
    split: str,
    transforms=None,
    mask_transforms=None,
) -> MvtecDataset:
    loader_cfg = config.get("loader", {})
    default_limit = _sanitize_limit(loader_cfg.get("max_samples"))
    use_full_train = bool(loader_cfg.get("use_full_train", False))
    if split == "train":
        limit = _sanitize_limit(loader_cfg.get("max_train_samples")) or default_limit
    else:
        limit = _sanitize_limit(loader_cfg.get("max_test_samples")) or default_limit

    image_transforms = transforms or _build_image_transforms(config)
    mask_transforms = mask_transforms or _build_mask_transforms(config)

    return MvtecDataset(
        category=category,
        split=split,
        root=config.get("root"),
        k_shot=loader_cfg.get("k_shot", 0),
        experiment_idx=loader_cfg.get("experiment_idx", 0),
        transforms=image_transforms,
        mask_transforms=mask_transforms,
        max_samples=limit,
        use_full_train_if_k_shot_zero=use_full_train,
    )


def build_visa_dataloaders(
    config: MutableMapping[str, Any],
    *,
    category: str,
    transforms=None,
    mask_transforms=None,
) -> Dict[str, Any]:
    loader_cfg = config.get("loader", {})
    k_shot = loader_cfg.get("k_shot", 0)
    experiment_idx = loader_cfg.get("experiment_idx", 0)
    root = config.get("root")
    split_type = config.get("split_type", "1cls")
    split_csv = config.get("split_csv")
    use_full_train = bool(loader_cfg.get("use_full_train", False))

    default_limit = _sanitize_limit(loader_cfg.get("max_samples"))
    train_limit = _sanitize_limit(loader_cfg.get("max_train_samples")) or default_limit
    test_limit = _sanitize_limit(loader_cfg.get("max_test_samples")) or default_limit
    test_file_numbers = loader_cfg.get("test_file_numbers")  # 특정 파일 번호 리스트

    image_transforms = transforms or _build_image_transforms(config)
    mask_transforms = mask_transforms or _build_mask_transforms(config)

    train_dataset = VisaDataset(
        category=category,
        split="train",
        root=root,
        split_type=split_type,
        split_csv=split_csv,
        k_shot=k_shot,
        experiment_idx=experiment_idx,
        use_full_train_if_k_shot_zero=use_full_train,
        transforms=image_transforms,
        mask_transforms=mask_transforms,
        max_samples=train_limit,
    )
    test_dataset = VisaDataset(
        category=category,
        split="test",
        root=root,
        split_type=split_type,
        split_csv=split_csv,
        k_shot=k_shot,
        experiment_idx=experiment_idx,
        transforms=image_transforms,
        mask_transforms=mask_transforms,
        max_samples=test_limit,
        test_file_numbers=test_file_numbers,
    )

    dataloader_cfg = {
        "batch_size": config.get("batch_size", 1),
        "num_workers": config.get("num_workers", 0),
        "pin_memory": config.get("pin_memory", False),
    }

    loaders = {
        "train": build_dataloader(train_dataset, dataloader_cfg, shuffle_override=False),
        "test": build_dataloader(test_dataset, dataloader_cfg, shuffle_override=False),
    }

    return {"datasets": {"train": train_dataset, "test": test_dataset}, "loaders": loaders}


def create_visa_dataset(
    config: MutableMapping[str, Any],
    *,
    category: str,
    split: str,
    transforms=None,
    mask_transforms=None,
) -> VisaDataset:
    loader_cfg = config.get("loader", {})
    default_limit = _sanitize_limit(loader_cfg.get("max_samples"))
    if split == "train":
        limit = _sanitize_limit(loader_cfg.get("max_train_samples")) or default_limit
    else:
        limit = _sanitize_limit(loader_cfg.get("max_test_samples")) or default_limit

    image_transforms = transforms or _build_image_transforms(config)
    mask_transforms = mask_transforms or _build_mask_transforms(config)

    return VisaDataset(
        category=category,
        split=split,
        root=config.get("root"),
        split_type=config.get("split_type", "1cls"),
        split_csv=config.get("split_csv"),
        k_shot=loader_cfg.get("k_shot", 0),
        experiment_idx=loader_cfg.get("experiment_idx", 0),
        transforms=image_transforms,
        mask_transforms=mask_transforms,
        max_samples=limit,
    )

