"""WinCLIP baseline integration."""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from .base import BaseMethod, ForwardResult, register_method
from methods.winclip_original.original import WinClipAD # cursor 수정 1108


@register_method("winclip")
class WinCLIPMethod(BaseMethod):
    """Wrapper around the original WinCLIP implementation."""

    def __init__(
        self,
        config: MutableMapping[str, Any],
        *,
        device: Optional[str] = None,
        normalize_score_maps: bool = True,
    ) -> None:
        super().__init__(config)
        self.device = torch.device(device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.normalize_score_maps = normalize_score_maps
        model_cfg = dict(config)
        self.backbone = model_cfg.get("backbone", "ViT-B-16-plus-240")
        self.pretrained_dataset = model_cfg.get("pretrained_dataset", "laion400m_e32")
        self.scales = tuple(model_cfg.get("scales", (2, 3)))
        self.precision = model_cfg.get("precision", "fp16")
        self.img_resize = model_cfg.get("img_resize", 240)
        self.img_cropsize = model_cfg.get("img_cropsize", 240)
        self.resolution = model_cfg.get("resolution", 400)
        self.k_shot = model_cfg.get("k_shot", 0)

        self._backend: Optional[WinClipAD] = None
        self._current_category: Optional[str] = None
        self._category_with_gallery: Dict[str, bool] = {}

    @classmethod
    def from_config(cls, config: MutableMapping[str, Any], **kwargs: Any) -> "WinCLIPMethod":
        return cls(
            config,
            device=kwargs.get("device"),
            normalize_score_maps=config.get("normalize_score_maps", True),
        )

    def setup(self) -> None:
        """Initialise WinCLIP backend."""
        if self._backend is not None:
            return

        self._backend = WinClipAD(
            out_size_h=self.resolution,
            out_size_w=self.resolution,
            device=str(self.device),
            backbone=self.backbone,
            pretrained_dataset=self.pretrained_dataset,
            scales=self.scales,
            precision=self.precision,
            img_resize=self.img_resize,
            img_cropsize=self.img_cropsize,
        )
        self._backend = self._backend.to(self.device)
        self._backend.eval_mode()

    def prepare_category(
        self,
        category: str,
        train_loader: Optional[DataLoader] = None,
    ) -> None:
        """Prepare text/image galleries for a given category."""
        self.setup()
        assert self._backend is not None

        if self._current_category != category:
            self._backend.build_text_feature_gallery(category)
            self._current_category = category

        if category in self._category_with_gallery:
            return

        if train_loader is None or self.k_shot == 0:
            self._backend.visual_gallery = None # cursor 추가
            self._category_with_gallery[category] = False
            return

        normal_images = []
        with torch.no_grad():
            for batch in train_loader:
                images = batch["image"]
                labels = batch["label"]
                normal_mask = labels == 0
                if normal_mask.sum() == 0:
                    continue
                for img in images[normal_mask]:
                    pil_img = to_pil_image(img.detach().cpu().clamp(0, 1))
                    normal_images.append(self._backend.transform(pil_img))

        if normal_images:
            stacked = torch.stack(normal_images, dim=0).to(self.device)
            self._backend.build_image_feature_gallery(stacked)
            self._category_with_gallery[category] = True
        else:
            self._backend.visual_gallery = None # cursor 추가
            self._category_with_gallery[category] = False

        self._backend.eval_mode()

    def forward(self, batch: Dict[str, Any]) -> ForwardResult:
        if self._backend is None:
            raise RuntimeError("WinCLIP backend not initialised. Call setup() first.")

        images = batch["image"].to(self.device)
        meta = batch.get("meta", {})

        if isinstance(meta, dict):
            categories = meta.get("category")
            if isinstance(categories, list):
                category = categories[0]
            else:
                category = categories
        else:
            category = None

        if isinstance(category, str):
            if self._current_category != category:
                self._backend.build_text_feature_gallery(category)
                self._current_category = category

        processed = [
            self._backend.transform(to_pil_image(img.detach().cpu().clamp(0, 1)))
            for img in images
        ]
        inputs = torch.stack(processed, dim=0).to(self.device)
        anomaly_maps = self._backend(inputs)  # list of numpy arrays
        pixel_scores_np = np.stack(anomaly_maps, axis=0).astype(np.float32)
        raw_pixel_scores = torch.from_numpy(pixel_scores_np)
        image_scores = raw_pixel_scores.view(raw_pixel_scores.size(0), -1).max(dim=1)[0]

        if self.normalize_score_maps:
            flat_scores = raw_pixel_scores.view(raw_pixel_scores.size(0), -1)
            min_vals = flat_scores.min(dim=1)[0].view(-1, 1, 1)
            max_vals = flat_scores.max(dim=1)[0].view(-1, 1, 1)
            denom = torch.clamp(max_vals - min_vals, min=1e-6)
            normalized_maps = (raw_pixel_scores - min_vals) / denom
        else:
            normalized_maps = None

        pixel_scores = raw_pixel_scores.unsqueeze(1)

        extra = {"raw_maps": pixel_scores_np}
        if normalized_maps is not None:
            extra["normalized_maps"] = normalized_maps.detach().cpu().numpy()

        return ForwardResult(
            image_scores=image_scores,
            pixel_scores=pixel_scores,
            extra=extra,
        )


__all__ = ["WinCLIPMethod"]

