"""Dataset base classes and DataLoader helpers."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, MutableMapping, Optional, TypeVar

from torch.utils.data import DataLoader, Dataset

T = TypeVar("T", bound="BaseDataset")


class BaseDataset(Dataset, metaclass=abc.ABCMeta):
    """모든 데이터셋 구현이 따라야 할 기본 인터페이스."""

    def __init__(self, transforms: Optional[Callable] = None) -> None:
        super().__init__()
        self.transforms = transforms

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Any:  # pragma: no cover - 인터페이스
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:  # pragma: no cover - 인터페이스
        raise NotImplementedError


@dataclass
class DataLoaderConfig:
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = False
    drop_last: bool = False


def build_dataloader(
    dataset: Dataset,
    loader_cfg: MutableMapping[str, Any],
    *,
    shuffle_override: Optional[bool] = None,
) -> DataLoader:
    """DataLoader 설정을 config에서 읽어 생성."""
    cfg = DataLoaderConfig(**loader_cfg)
    shuffle = shuffle_override if shuffle_override is not None else cfg.shuffle
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
    )


__all__ = [
    "BaseDataset",
    "build_dataloader",
    "DataLoaderConfig",
]

