"""AF-CLIP TINY method integration."""

from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, MutableMapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseMethod, ForwardResult, register_method

LOGGER = logging.getLogger(__name__)

AF_CLIP_TINY_ROOT = Path(__file__).resolve().parents[2] / "git_clone" / "AF-CLIP_tiny"
if not AF_CLIP_TINY_ROOT.exists():
    raise ImportError(
        "AF-CLIP TINY 서브모듈이 존재하지 않습니다. "
        "git_clone/AF-CLIP_tiny 경로를 확인하세요."
    )

AF_CLIP_TINY_PATH_STR = str(AF_CLIP_TINY_ROOT)
if AF_CLIP_TINY_PATH_STR not in sys.path:
    sys.path.insert(0, AF_CLIP_TINY_PATH_STR)

try:
    from model.clip import CLIP  # type: ignore
    from model.tokenizer import tokenize  # type: ignore
except ImportError as exc:  # pragma: no cover - 환경 의존
    raise ImportError(
        "AF-CLIP TINY 모듈을 import 할 수 없습니다. "
        "필요한 의존성이 설치되어 있는지 확인하세요."
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


def load_model(path: str, args: SimpleNamespace, device: torch.device):
    """AF-CLIP TINY 모델 로드 함수 (main.py에서 가져옴)."""
    # 가중치 불러오기
    ckpt = torch.load(path, map_location="cpu")

    if 'state_dict' in ckpt:
        checkpoint = ckpt['state_dict']
    else:
        checkpoint = ckpt

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_k = k
        if k.startswith('_image_encoder.module.'):
            new_k = k.removeprefix('_image_encoder.module.')
        elif k.startswith('_text_encoder.module.'):
            new_k = k.removeprefix('_text_encoder.module.')
        elif k.startswith('_logit_scale.module.'):
            new_k = k.removeprefix('_logit_scale.module.')
        
        new_state_dict[new_k] = v

    state_dict = new_state_dict

    vocab_size = state_dict["token_embedding.weight"].shape[0]
    context_length = state_dict["positional_embedding"].shape[0]

    # projection 후 임베딩 차원 (공통 차원)
    embed_dim = state_dict["text_projection"].shape[1]
    # ViT 임베딩 차원
    vision_embed_dim = state_dict["visual.conv1.weight"].shape[0]
    # 트랜스포머 임베딩 차원
    text_embed_dim = state_dict["ln_final.weight"].shape[0]

    # ViT layer 수
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    # 트랜스포머 layer 수
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # ViT 헤드
    vision_heads = vision_embed_dim // 64
    # 트랜스포머 헤드
    transformer_heads = text_embed_dim // 64

    # ViT 패치 크기
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    # ViT grid 크기 (한 변의 패치 개수)
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    # 이미지 해상도
    image_resolution = vision_patch_size * grid_size

    model = CLIP(
        args=args,
        embed_dim=embed_dim,
        # vision
        vision_embed_dim=vision_embed_dim,
        vision_heads=vision_heads, 
        vision_layers=vision_layers,
        image_resolution=image_resolution,
        vision_patch_size=vision_patch_size,
        # text
        text_embed_dim=text_embed_dim,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers, 
        context_length=context_length,
        vocab_size=vocab_size,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    model.load_state_dict(state_dict)

    if str(device) == "cpu":
        model.float()

    return model.eval()


@register_method("afclip_tiny")
class AFClipTinyMethod(BaseMethod):
    """Wrapper around the AF-CLIP TINY implementation."""

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

        self.clip_weight_path: str = config.get("clip_weight", "./weight/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt")
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
        self.img_size: int = int(config.get("img_size", 518))

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
    ) -> "AFClipTinyMethod":
        return cls(config, device=kwargs.get("device"))

    def setup(self) -> None:
        if self._model is not None:
            return

        # 경로 처리: 상대 경로인 경우 프로젝트 루트 기준으로 변환
        clip_weight_path = Path(self.clip_weight_path)
        if not clip_weight_path.is_absolute():
            # 상대 경로인 경우 AF-CLIP_tiny 루트 기준으로 해석
            clip_weight_path = AF_CLIP_TINY_ROOT / clip_weight_path
        else:
            clip_weight_path = Path(self.clip_weight_path)

        if not clip_weight_path.exists():
            raise FileNotFoundError(
                f"CLIP 가중치 파일을 찾을 수 없습니다: {clip_weight_path}"
            )

        clip_model = load_model(
            str(clip_weight_path),
            self._args,
            self.device,
        )

        # 이미지 크기 조정을 위한 transform 설정
        # main.py의 로직을 참고하여 transform을 적용하지 않고 직접 처리
        clip_model = clip_model.to(self.device)
        clip_model.eval()

        # CLIP 모델의 모든 파라미터를 고정
        for param in clip_model.parameters():
            param.requires_grad_(False)

        # 기존 CLIP에 새로운 모듈 Adaptor와 Prompt를 삽입
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
                "AF-CLIP TINY 사전 학습 가중치를 찾을 수 없습니다: "
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

        model.state_prompt_embedding = prompt_tensor.to(self.device)
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
            # detect_forward 사용 (memory bank가 있으면 자동으로 결합)
            cls_label, predict_map = self._model.detect_forward(images, self._args)

        # 이미지 레벨 점수는 cls_label 사용
        image_scores = cls_label

        if self.interpolate_to_input:
            pixel_maps = F.interpolate(
                predict_map,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            pixel_maps = predict_map

        if self.apply_blur and self._blur_layer is not None:
            pixel_maps = self._blur_layer(pixel_maps)

        image_scores = image_scores.to(torch.float32)
        pixel_maps = pixel_maps.to(torch.float32)

        return ForwardResult(
            image_scores=image_scores,
            pixel_scores=pixel_maps,
        )


__all__ = ["AFClipTinyMethod"]

