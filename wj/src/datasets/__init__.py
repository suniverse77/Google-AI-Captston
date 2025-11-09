"""Dataset helper functions."""

from __future__ import annotations

from typing import Any, Dict, MutableMapping

from torch.utils.data import DataLoader

from .base import build_dataloader
from .mvtec import MvtecDataset

__all__ = ["build_mvtec_dataloaders", "create_mvtec_dataset"]


def _sanitize_limit(value):
    if value is None:
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    return value_int if value_int > 0 else None


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

    default_limit = _sanitize_limit(loader_cfg.get("max_samples"))
    train_limit = _sanitize_limit(loader_cfg.get("max_train_samples")) or default_limit
    test_limit = _sanitize_limit(loader_cfg.get("max_test_samples")) or default_limit

    train_dataset = MvtecDataset(
        category=category,
        split="train",
        root=root,
        k_shot=k_shot,
        experiment_idx=experiment_idx,
        transforms=transforms,
        mask_transforms=mask_transforms,
        max_samples=train_limit,
    )
    test_dataset = MvtecDataset(
        category=category,
        split="test",
        root=root,
        k_shot=k_shot,
        experiment_idx=experiment_idx,
        transforms=transforms,
        mask_transforms=mask_transforms,
        max_samples=test_limit,
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
    if split == "train":
        limit = _sanitize_limit(loader_cfg.get("max_train_samples")) or default_limit
    else:
        limit = _sanitize_limit(loader_cfg.get("max_test_samples")) or default_limit

    return MvtecDataset(
        category=category,
        split=split,
        root=config.get("root"),
        k_shot=loader_cfg.get("k_shot", 0),
        experiment_idx=loader_cfg.get("experiment_idx", 0),
        transforms=transforms,
        mask_transforms=mask_transforms,
        max_samples=limit,
    )

