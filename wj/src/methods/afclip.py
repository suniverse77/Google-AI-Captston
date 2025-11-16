"""AF-CLIP method integration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, MutableMapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseMethod, ForwardResult, register_method

LOGGER = logging.getLogger(__name__)

AF_CLIP_ROOT = Path(__file__).resolve().parents[2] / "git_clone" / "AF-CLIP"
if not AF_CLIP_ROOT.exists():
    raise ImportError(
        "AF-CLIP 서브모듈이 존재하지 않습니다. "
        "git_clone/AF-CLIP 경로를 확인하세요."
    )

AF_CLIP_PATH_STR = str(AF_CLIP_ROOT)
if AF_CLIP_PATH_STR not in sys.path:
    sys.path.insert(0, AF_CLIP_PATH_STR)

try:
    from clip.clip import load as clip_load, tokenize  # type: ignore
except ImportError as exc:  # pragma: no cover - 환경 의존
    raise ImportError(
        "AF-CLIP 모듈을 import 할 수 없습니다. "
        "`pip install -r git_clone/AF-CLIP/requirements.txt` 명령으로 "
        "의존성을 설치했는지 확인하세요."
    ) from exc


def _to_list(value: Sequence[int] | int) -> Sequence[int]:
    if isinstance(value, (list, tuple)):
        return value
    return [int(value)]


def _load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # For older PyTorch versions without the weights_only argument
        return torch.load(path, map_location=device)


@register_method("afclip")
class AFClipMethod(BaseMethod):
    """Wrapper around the AF-CLIP implementation."""

    def __init__(
        self,
        config: MutableMapping[str, Any],
        *,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        device_str = device or config.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(device_str)

        self.model_name: str = config.get("clip_model", "ViT-L/14@336px")
        self.download_root: Optional[str] = config.get("clip_download_dir")
        self.prompt_len: int = int(config.get("prompt_len", 12))
        self.feature_layers: Sequence[int] = list(
            map(int, _to_list(config.get("feature_layers", [6, 12, 18, 24])))
        )
        self.memory_layers: Sequence[int] = list(
            map(int, _to_list(config.get("memory_layers", [6, 12, 18, 24])))
        )
        self.alpha: float = float(config.get("alpha", 0.1))
        self.fewshot: int = int(config.get("fewshot", 0))
        self.interpolate_to_input: bool = bool(
            config.get("interpolate_to_input", True)
        )

        blur_cfg = config.get("gaussian_blur", {})
        self.apply_blur: bool = bool(blur_cfg.get("enabled", False))
        self.blur_kernel: int = int(blur_cfg.get("kernel_size", 5))
        self.blur_sigma: float = float(blur_cfg.get("sigma", 4.0))
        self._blur_layer: Optional[torch.nn.Module] = None

        weight_cfg = config.get("weight", {})
        self.weight_dir: Optional[Path] = (
            Path(weight_cfg["dir"]) if weight_cfg.get("dir") else None
        )
        self.weight_dataset: str = weight_cfg.get("dataset", "mvtec")
        self.weight_prefix: Optional[str] = weight_cfg.get("prefix")
        self.weight_strict: bool = bool(weight_cfg.get("strict", False))

        self._args = SimpleNamespace(
            prompt_len=self.prompt_len,
            feature_layers=list(self.feature_layers),
            memory_layers=list(self.memory_layers),
            alpha=self.alpha,
        )

        self._model = None
        self._current_category: Optional[str] = None

    @classmethod
    def from_config(
        cls,
        config: MutableMapping[str, Any],
        **kwargs: Any,
    ) -> "AFClipMethod":
        return cls(config, device=kwargs.get("device"))

    def setup(self) -> None:
        if self._model is not None:
            return

        clip_model, _ = clip_load(
            self.model_name,
            device=str(self.device),
            download_root=self.download_root,
            jit=False,
        )
        clip_model = clip_model.to(self.device)
        clip_model.eval()

        clip_model.insert(args=self._args, tokenizer=tokenize, device=self.device)
        self._load_pretrained_weights(clip_model)

        if self.apply_blur:
            if hasattr(torch.nn, "GaussianBlur"):
                blur_layer = torch.nn.GaussianBlur(
                    kernel_size=self.blur_kernel,
                    sigma=self.blur_sigma,
                )
                self._blur_layer = blur_layer.to(self.device)
            else:
                LOGGER.warning(
                    "GaussianBlur 모듈을 사용할 수 없어 블러 옵션을 비활성화합니다."
                )
                self.apply_blur = False
                self._blur_layer = None

        self._model = clip_model

    def _load_pretrained_weights(self, model) -> None:
        if self.weight_dir is None:
            return

        prefix = self.weight_prefix or self.weight_dataset
        prompt_path = self.weight_dir / f"{prefix}_prompt.pt"
        adaptor_path = self.weight_dir / f"{prefix}_adaptor.pt"

        missing_files = [
            path for path in [prompt_path, adaptor_path] if not path.exists()
        ]
        if missing_files:
            message = (
                "AF-CLIP 사전 학습 가중치를 찾을 수 없습니다: "
                + ", ".join(str(path) for path in missing_files)
            )
            if self.weight_strict:
                raise FileNotFoundError(message)
            LOGGER.warning(message)
            return

        prompt_loaded = _load_checkpoint(prompt_path, self.device)
        adaptor_loaded = _load_checkpoint(adaptor_path, self.device)

        if isinstance(prompt_loaded, torch.nn.Module):
            prompt_tensor = next(prompt_loaded.parameters()).detach()
        elif isinstance(prompt_loaded, (torch.Tensor, torch.nn.Parameter)):
            prompt_tensor = prompt_loaded.detach()
        elif isinstance(prompt_loaded, dict):
            # 일부 체크포인트는 {"state_prompt_embedding": tensor} 형태
            if "state_prompt_embedding" in prompt_loaded:
                prompt_tensor = prompt_loaded["state_prompt_embedding"].detach()
            else:
                raise TypeError(
                    f"지원하지 않는 프롬프트 체크포인트 dict 키입니다: {prompt_loaded.keys()}"
                )
        else:
            raise TypeError(
                f"지원하지 않는 프롬프트 체크포인트 형식입니다: {type(prompt_loaded)}"
            )

        if isinstance(adaptor_loaded, torch.nn.Module):
            adaptor_state = adaptor_loaded.state_dict()
        elif isinstance(adaptor_loaded, dict):
            adaptor_state = adaptor_loaded
        else:
            raise TypeError(
                f"지원하지 않는 어댑터 체크포인트 형식입니다: {type(adaptor_loaded)}"
            )

        # state_prompt_embedding은 nn.Parameter이므로 데이터만 업데이트
        prompt_tensor = prompt_tensor.to(self.device)
        if model.state_prompt_embedding.shape != prompt_tensor.shape:
            raise ValueError(
                f"프롬프트 임베딩 크기가 일치하지 않습니다: "
                f"모델 {model.state_prompt_embedding.shape} vs 로드된 {prompt_tensor.shape}"
            )
        model.state_prompt_embedding.data.copy_(prompt_tensor)
        model.state_prompt_embedding.requires_grad_(False)

        model.adaptor.load_state_dict(adaptor_state)
        model.adaptor = model.adaptor.to(self.device)
        model.adaptor.eval()

    def _prepare_memory(self, train_loader: DataLoader) -> None:
        assert self._model is not None
        if self.fewshot <= 0:
            self._model.memorybank = None
            return

        collected: list[torch.Tensor] = []
        needed = self.fewshot

        with torch.no_grad():
            for batch in train_loader:
                images = batch["image"].to(self.device)
                labels = batch.get("label")

                if labels is None:
                    normal_mask = torch.ones(images.size(0), dtype=torch.bool, device=self.device)
                else:
                    if not torch.is_tensor(labels):
                        labels_tensor = torch.tensor(labels, device=self.device)
                    else:
                        labels_tensor = labels.to(self.device)
                    normal_mask = labels_tensor == 0

                if normal_mask.sum() == 0:
                    continue

                normal_images = images[normal_mask]
                for img in normal_images:
                    collected.append(img.detach())
                    if len(collected) >= needed:
                        break
                if len(collected) >= needed:
                    break

        if not collected:
            LOGGER.warning(
                "카테고리 메모리 생성을 위한 정상 샘플을 찾지 못했습니다. fewshot=%s",
                self.fewshot,
            )
            self._model.memorybank = None
            return

        stack = torch.stack(collected[:needed], dim=0).to(self.device)
        self._model.store_memory(stack, self._args)

    def prepare_category(
        self,
        category: str,
        train_loader: Optional[DataLoader] = None,
    ) -> None:
        self.setup()
        assert self._model is not None

        if self._current_category == category:
            return

        self._model.memorybank = None
        if train_loader is not None:
            self._prepare_memory(train_loader)
        else:
            self._model.memorybank = None

        self._current_category = category

    def forward(self, batch: Dict[str, Any]) -> ForwardResult:
        self.setup()
        assert self._model is not None

        images = batch["image"].to(self.device)

        with torch.no_grad():
            image_scores, pixel_maps = self._model.detect_forward(images, self._args)

        if self.interpolate_to_input:
            pixel_maps = F.interpolate(
                pixel_maps,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.apply_blur and self._blur_layer is not None:
            pixel_maps = self._blur_layer(pixel_maps)

        image_scores = image_scores.to(torch.float32)
        pixel_maps = pixel_maps.to(torch.float32)

        return ForwardResult(
            image_scores=image_scores,
            pixel_scores=pixel_maps,
        )


__all__ = ["AFClipMethod"]

