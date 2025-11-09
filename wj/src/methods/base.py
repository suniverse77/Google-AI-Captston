"""모델 메서드 인터페이스 및 레지스트리."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, MutableMapping, Optional, Type, TypeVar

import torch

T = TypeVar("T", bound="BaseMethod")


@dataclass
class ForwardResult:
    image_scores: torch.Tensor  # (B,)
    pixel_scores: Optional[torch.Tensor] = None  # (B, H, W)
    embeddings: Optional[torch.Tensor] = None
    extra: Optional[Dict[str, Any]] = None


class BaseMethod(torch.nn.Module, metaclass=abc.ABCMeta):
    """WinCLIP / TinyCLIP 등 공통 인터페이스."""

    def __init__(self, config: MutableMapping[str, Any]) -> None:
        super().__init__()
        self.config = config

    @classmethod
    @abc.abstractmethod
    def from_config(cls: Type[T], config: MutableMapping[str, Any], **kwargs: Any) -> T:
        """YAML config 기반 인스턴스 생성."""

    @abc.abstractmethod
    def setup(self) -> None:
        """사전 학습 가중치 로드 등 초기화 로직."""

    @abc.abstractmethod
    def forward(self, batch: Dict[str, Any]) -> ForwardResult:
        """단일 배치 추론."""

    def predict(self, batch: Dict[str, Any]) -> ForwardResult:
        self.eval()
        with torch.no_grad():
            return self.forward(batch)


METHOD_REGISTRY: Dict[str, Type[BaseMethod]] = {}


def register_method(name: str) -> Any:
    """메서드 클래스를 registry에 등록."""

    def decorator(cls: Type[T]) -> Type[T]:
        if name in METHOD_REGISTRY:
            raise KeyError(f"이미 등록된 메서드입니다: {name}")
        METHOD_REGISTRY[name] = cls
        cls.NAME = name  # type: ignore[attr-defined]
        return cls

    return decorator


def build_method(config: MutableMapping[str, Any], **kwargs: Any) -> BaseMethod:
    method_name = config.get("name")
    if method_name is None:
        raise KeyError("model.name 설정이 필요합니다.")
    if method_name not in METHOD_REGISTRY:
        raise KeyError(f"등록되지 않은 메서드입니다: {method_name}")

    method_cls = METHOD_REGISTRY[method_name]
    model = method_cls.from_config(config, **kwargs)
    model.setup()
    return model


__all__ = [
    "BaseMethod",
    "ForwardResult",
    "register_method",
    "build_method",
]

