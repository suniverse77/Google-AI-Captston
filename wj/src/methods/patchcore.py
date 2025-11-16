"""PatchCore integration wrapper."""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import BaseMethod, ForwardResult, register_method

LOGGER = logging.getLogger(__name__)

PATCHCORE_SRC = (
    Path(__file__).resolve().parents[2]
    / "git_clone"
    / "patchcore-inspection"
    / "src"
)
if PATCHCORE_SRC.exists():
    patchcore_path = str(PATCHCORE_SRC)
    if patchcore_path not in sys.path:
        sys.path.insert(0, patchcore_path)

try:
    import patchcore.backbones  # type: ignore
    import patchcore.common  # type: ignore
    import patchcore.patchcore as patchcore_lib  # type: ignore
    import patchcore.sampler  # type: ignore
except ImportError as exc:  # pragma: no cover - 환경 문제
    raise ImportError(
        "patchcore 라이브러리를 찾을 수 없습니다. "
        "`git_clone/patchcore-inspection` 경로가 존재하고 "
        "`pip install -r git_clone/patchcore-inspection/requirements.txt`로 "
        "의존성을 설치했는지 확인하세요."
    ) from exc


def _to_input_shape(value: Sequence[int]) -> Sequence[int]:
    if len(value) != 3:
        raise ValueError("input_size는 [C, H, W] 형태여야 합니다.")
    return value


def _normalize_score_maps(scores: np.ndarray) -> np.ndarray:
    flat = scores.reshape(scores.shape[0], -1)
    min_vals = flat.min(axis=1, keepdims=True)
    max_vals = flat.max(axis=1, keepdims=True)
    denom = np.maximum(max_vals - min_vals, 1e-6)
    normalized = (flat - min_vals) / denom
    return normalized.reshape(scores.shape)


@register_method("patchcore")
class PatchCoreMethod(BaseMethod):
    """Wrapper around the original PatchCore implementation."""

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

        model_cfg = dict(config)
        self.backbone_name: str = model_cfg.get("backbone", "resnet18")
        self.layers_to_extract: Sequence[str] = tuple(
            model_cfg.get("layers", ["layer2", "layer3"])
        )
        self.input_shape: Sequence[int] = _to_input_shape(
            model_cfg.get("input_size", [3, 224, 224])
        )
        self.pretrain_embed_dim: int = int(model_cfg.get("pretrain_embed_dim", 1024))
        self.target_embed_dim: int = int(model_cfg.get("target_embed_dim", 384))

        patch_cfg = model_cfg.get("patch", {})
        self.patch_size: int = int(patch_cfg.get("size", 3))
        self.patch_stride: int = int(patch_cfg.get("stride", 1))

        nn_cfg = model_cfg.get("nn", {})
        self.nn_k: int = int(nn_cfg.get("k", 5))
        self.faiss_on_gpu: bool = bool(nn_cfg.get("faiss_on_gpu", False))
        self.faiss_num_workers: int = int(nn_cfg.get("num_workers", 4))

        sampler_cfg = model_cfg.get("sampler", {})
        self.sampler_name: str = str(sampler_cfg.get("name", "identity")).lower()
        self.sampler_ratio: float = float(sampler_cfg.get("ratio", 0.1))

        pretrained_cfg = model_cfg.get("pretrained") or {}
        pretrained_root = pretrained_cfg.get("root")
        self.pretrained_root: Optional[Path] = (
            Path(pretrained_root) if pretrained_root else None
        )
        self.pretrained_prepend: str = str(pretrained_cfg.get("prepend", ""))
        self.pretrained_strict: bool = bool(pretrained_cfg.get("strict", False))

        score_cfg = model_cfg.get("score_map", {})
        self.normalize_maps: bool = bool(score_cfg.get("normalize", True))

        self._patchcore: Optional[patchcore_lib.PatchCore] = None
        self._current_category: Optional[str] = None
        self._nn_method: Optional[patchcore.common.FaissNN] = None
        self._prepend_cache: Dict[str, str] = {}

    @classmethod
    def from_config(
        cls,
        config: MutableMapping[str, Any],
        **kwargs: Any,
    ) -> "PatchCoreMethod":
        return cls(config, device=kwargs.get("device"))

    def setup(self) -> None:
        if self._patchcore is not None:
            return

        backbone = patchcore.backbones.load(self.backbone_name)
        backbone.name = self.backbone_name

        sampler = self._build_sampler(self.sampler_name, self.sampler_ratio)
        self._nn_method = patchcore.common.FaissNN(
            self.faiss_on_gpu, self.faiss_num_workers
        )

        patchcore_model = patchcore_lib.PatchCore(self.device)
        patchcore_model.load(
            backbone=backbone,
            layers_to_extract_from=self.layers_to_extract,
            device=self.device,
            input_shape=tuple(self.input_shape),
            pretrain_embed_dimension=self.pretrain_embed_dim,
            target_embed_dimension=self.target_embed_dim,
            patchsize=self.patch_size,
            patchstride=self.patch_stride,
            anomaly_score_num_nn=self.nn_k,
            featuresampler=sampler,
            nn_method=self._nn_method,
        )
        self._patchcore = patchcore_model

    def _build_sampler(
        self,
        name: str,
        ratio: float,
    ) -> Any:
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        if name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(ratio, self.device)
        if name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(
                ratio,
                self.device,
            )
        if name == "random":
            return patchcore.sampler.RandomSampler(ratio)
        raise ValueError(f"지원되지 않는 sampler입니다: {name}")

    def prepare_category(
        self,
        category: str,
        train_loader: Optional[DataLoader] = None,
    ) -> None:
        self.setup()
        if self._patchcore is None:
            raise RuntimeError("PatchCore 모델이 초기화되지 않았습니다.")
        if category == self._current_category:
            return
        if self._try_load_pretrained(category):
            self._current_category = category
            return

        if train_loader is None:
            raise ValueError("PatchCore는 train_loader가 필요합니다.")

        self._patchcore.fit(train_loader)
        self._current_category = category

    def _try_load_pretrained(self, category: str) -> bool:
        if self.pretrained_root is None:
            return False
        if self._patchcore is None:
            return False
        category_dir = self.pretrained_root / "models" / f"mvtec_{category}"
        if not category_dir.exists():
            message = (
                "사전 학습 PatchCore 경로를 찾을 수 없습니다: %s", category_dir
            )
            if self.pretrained_strict:
                raise FileNotFoundError(message[0] % message[1])
            LOGGER.warning(*message)
            return False

        prepend = self._resolve_prepend(category, category_dir)
        LOGGER.info(
            "사전 학습 PatchCore 로드: category=%s, path=%s, prepend=%s",
            category,
            category_dir,
            prepend,
        )
        nn_method = self._nn_method or patchcore.common.FaissNN(
            self.faiss_on_gpu, self.faiss_num_workers
        )
        self._nn_method = nn_method
        self._patchcore.load_from_path(
            str(category_dir),
            device=self.device,
            nn_method=nn_method,
            prepend=prepend,
        )
        return True

    def _resolve_prepend(self, category: str, category_dir: Path) -> str:
        if self.pretrained_prepend:
            return self.pretrained_prepend
        if category in self._prepend_cache:
            return self._prepend_cache[category]

        candidates = list(category_dir.glob("*_patchcore_params.pkl"))
        if not candidates:
            return ""
        selected = sorted(candidates)[0]
        prepend = selected.name.replace("patchcore_params.pkl", "")
        self._prepend_cache[category] = prepend
        return prepend

    def forward(self, batch: Dict[str, Any]) -> ForwardResult:
        self.setup()
        if self._patchcore is None:
            raise RuntimeError("PatchCore 모델이 초기화되지 않았습니다.")

        images = batch["image"].to(self.device)
        scores_list, masks_list = self._patchcore.predict(images)

        scores_tensor = torch.tensor(scores_list, dtype=torch.float32, device=self.device)
        pixel_scores_tensor = None
        extra: Dict[str, Any] = {"raw_scores": scores_list}

        if masks_list:
            masks_np = np.stack(masks_list, axis=0).astype(np.float32)
            extra["raw_maps"] = masks_np.copy()
            if self.normalize_maps:
                normalized_np = _normalize_score_maps(masks_np)
                extra["normalized_maps"] = normalized_np.copy()
                masks_np = normalized_np
            pixel_scores_tensor = torch.from_numpy(masks_np).unsqueeze(1).to(self.device)

        return ForwardResult(
            image_scores=scores_tensor,
            pixel_scores=pixel_scores_tensor,
            extra=extra,
        )


__all__ = ["PatchCoreMethod"]

