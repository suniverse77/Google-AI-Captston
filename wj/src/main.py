"""WinCLIP baseline 실험 진입점."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import math
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import (
    build_afclip_mvtec_dataloaders,
    build_mvtec_dataloaders,
    build_visa_dataloaders,
)
from datasets.base import build_dataloader
from eval.metrics import summarize_metrics, save_metrics_csv
from eval.utils import specify_resolution
from eval.latency import LatencyConfig, profile_model
from methods.base import ForwardResult, build_method
from utils import config as config_utils
from utils.logger import create_experiment_dir, setup_logger
from utils.prompts import get_prompt_templates
from utils.seed import set_seed
from vis.overlay import save_overlay_examples

import methods.winclip  # 등록 사이드이펙트용 (cursor 1108)
import methods.patchcore  # PatchCore 등록
import methods.afclip  # AF-CLIP 등록
import methods.afclip_tiny  # AF-CLIP TINY 등록


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WinCLIP baseline on MVTec-AD.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvtec_winclip.yaml",
        help="실험 설정 YAML 경로 (프로젝트 루트 기준).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="강제로 사용할 디바이스 지정 (예: cpu, cuda:0)",
    )
    return parser.parse_args()


DATASET_BUILDERS = {
    "mvtec": build_mvtec_dataloaders,
    "visa": build_visa_dataloaders,
    "mvtec_afclip": build_afclip_mvtec_dataloaders,
}


def _create_denormalizer(normalization_cfg: Optional[Dict[str, Any]]) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    if not normalization_cfg:
        return None

    mean = normalization_cfg.get("mean")
    std = normalization_cfg.get("std")
    if mean is None or std is None:
        return None

    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def _denormalize(image: torch.Tensor) -> torch.Tensor:
        img = image.clone()
        img.mul_(std_tensor).add_(mean_tensor)
        return torch.clamp(img, 0.0, 1.0)

    return _denormalize


def run_category(
    cfg: Dict[str, Any],
    *,
    category: str,
    output_root: Path,
    logger,
) -> Dict[str, float]:
    logger.info("=== Category: %s ===", category)

    dataset_name = cfg["data"].get("dataset", "mvtec")
    builder = DATASET_BUILDERS.get(dataset_name)
    if builder is None:
        raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")

    dataloaders = builder(
        cfg["data"],
        category=category,
        transforms=None,
        mask_transforms=None,
    )
    loaders = dataloaders["loaders"]
    train_loader: DataLoader = loaders["train"]
    test_loader: DataLoader = loaders["test"]

    model_cfg = dict(cfg["model"])
    if cfg.get("model", {}).get("prompt_template"):
        templates = get_prompt_templates(cfg["model"]["prompt_template"])
        model_cfg["prompt_templates"] = templates
    if cfg.get("runtime", {}).get("deterministic", True):
        torch.set_grad_enabled(False)

    model = build_method(model_cfg, device=cfg["model"].get("device"))

    model.prepare_category(category, train_loader)

    image_scores: List[float] = []
    image_labels: List[int] = []
    pixel_scores_batches: List[np.ndarray] = []
    pixel_labels_batches: List[np.ndarray] = []
    vis_cfg = cfg.get("visualization", {})
    overlay_limit = int(vis_cfg.get("num_examples", 0))
    save_overlay = bool(vis_cfg.get("save_overlay", False))
    overlay_samples: List[Dict[str, Any]] = []

    denormalize_fn = _create_denormalizer(cfg.get("data", {}).get("transforms", {}).get("normalization"))

    for batch_idx, batch in enumerate(test_loader):
        result: ForwardResult = model.predict(batch)

        scores = result.image_scores.detach().cpu().numpy().tolist()
        labels = (
            batch["label"].detach().cpu().numpy().tolist()
            if hasattr(batch["label"], "detach")
            else batch["label"]
        )

        image_scores.extend(scores)
        image_labels.extend(labels)

        if result.pixel_scores is not None:
            pixel_scores_batches.append(result.pixel_scores.detach().cpu().numpy())
            if batch.get("mask") is not None:
                mask = batch["mask"]
                mask_np = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
                pixel_labels_batches.append(mask_np)

        if save_overlay and overlay_limit and len(overlay_samples) < overlay_limit:
            image_tensor = batch["image"][0]
            if denormalize_fn is not None:
                image_tensor = denormalize_fn(image_tensor.cpu())
            overlay_mask = batch.get("mask")
            # 이미지 경로에서 파일 번호 추출
            image_path = None
            file_number = None
            if batch.get("meta") is not None:
                if isinstance(batch["meta"], (list, tuple)) and len(batch["meta"]) > 0:
                    meta = batch["meta"][0]
                    if isinstance(meta, dict):
                        image_path = meta.get("path", "")
                elif isinstance(batch["meta"], dict):
                    image_path = batch["meta"].get("path", "")
            
            # image_path가 리스트인 경우 첫 번째 요소 사용
            if isinstance(image_path, (list, tuple)) and len(image_path) > 0:
                image_path = image_path[0]
            
            if image_path and isinstance(image_path, str):
                # 파일명에서 숫자 추출 (예: "000.JPG" -> 0, "001.JPG" -> 1)
                filename = Path(image_path).stem
                match = re.search(r'(\d+)', filename)
                if match:
                    file_number = int(match.group(1))
            
            overlay_samples.append(
                {
                    "image": image_tensor,
                    "mask": overlay_mask[0].cpu() if overlay_mask is not None else None,
                    "heatmap": result.pixel_scores[0] if result.pixel_scores is not None else None,
                    "file_number": file_number,  # 실제 파일 번호 저장
                    "image_path": image_path,  # 디버깅용 경로 저장
                    "titles": [
                        "Image",
                        "Mask",
                        "Heatmap",
                        "Overlay",
                    ],
                }
            )

    pixel_scores = np.concatenate(pixel_scores_batches) if pixel_scores_batches else None
    pixel_labels = np.concatenate(pixel_labels_batches) if pixel_labels_batches else None

    if pixel_scores is not None and pixel_labels is not None:
        resolution = cfg["model"].get("resolution", 400)
        score_maps = [np.squeeze(s) for s in pixel_scores]
        mask_maps = [np.squeeze(m) for m in pixel_labels]
        _, resized_scores, resized_masks = specify_resolution(
            image_list=[np.zeros_like(score_maps[0]) for _ in range(len(score_maps))],
            score_list=score_maps,
            mask_list=mask_maps,
            resolution=(resolution, resolution),
        )
        pixel_scores = np.stack(resized_scores, axis=0)
        pixel_labels = np.stack(resized_masks, axis=0)

    pro_cfg = cfg.get("evaluation", {}).get("metrics", {}).get("pro", {})
    metrics = summarize_metrics(
        image_scores,
        image_labels,
        pixel_scores=pixel_scores,
        pixel_labels=pixel_labels,
        pro_cfg=pro_cfg,
    )

    # Latency 측정 (metrics에 추가하기 위해 save_metrics_csv 호출 전에 수행)
    latency_cfg = cfg.get("evaluation", {}).get("latency", {})
    if latency_cfg:
        # Latency 측정은 batch size 1로 수행 (논문 기준)
        test_dataset = dataloaders["datasets"]["test"]
        latency_loader_cfg = {
            "batch_size": 1,
            "num_workers": cfg["data"].get("num_workers", 0),
            "pin_memory": cfg["data"].get("pin_memory", False),
        }
        latency_loader = build_dataloader(test_dataset, latency_loader_cfg, shuffle_override=False)
        
        # Throughput 측정은 batch size 16으로 수행 (논문 기준)
        throughput_loader_cfg = {
            "batch_size": 16,
            "num_workers": cfg["data"].get("num_workers", 0),
            "pin_memory": cfg["data"].get("pin_memory", False),
        }
        throughput_loader = build_dataloader(test_dataset, throughput_loader_cfg, shuffle_override=False)
        
        latency_stats = profile_model(
            model,
            latency_loader,
            latency_cfg=LatencyConfig(
                warmup_iters=latency_cfg.get("warmup_iters", 5),
                measure_iters=latency_cfg.get("measure_iters", 20),
                repeat=latency_cfg.get("repeat", 3),
            ),
            throughput_inputs=throughput_loader,
            throughput_num_passes=latency_cfg.get("throughput_num_passes", 1000),
            gpu_memory_num_iters=latency_cfg.get("gpu_memory_num_iters", 1000),
        )
        metrics.update(latency_stats)

    category_dir = output_root / category
    category_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_csv(metrics, category_dir / "metrics.csv")

    if save_overlay and overlay_limit and overlay_samples:
        filtered_samples = [
            s for s in overlay_samples if s["heatmap"] is not None
        ]
        if filtered_samples:
            save_overlay_examples(
                filtered_samples,
                output_dir=category_dir / "overlays",
                cmap=vis_cfg.get("cmap", "plasma"),
                alpha=vis_cfg.get("overlay_alpha", 0.5),
            )

    return metrics


def run(cfg: Dict[str, Any]) -> None:
    seed = cfg["experiment"].get("seed", 2025)
    set_seed(seed, deterministic=cfg.get("runtime", {}).get("deterministic", True))

    log_dir = Path(cfg["experiment"].get("output_root", "results/mvtec"))
    experiment_dir = create_experiment_dir(
        log_dir,
        cfg["experiment"].get("name", "mvtec_experiment"),
    )
    config_utils.save_config_copy(cfg, experiment_dir, filename="config_used.yaml")

    logger = setup_logger("winclip", log_dir=experiment_dir)
    logger.info("Experiment dir: %s", experiment_dir)

    categories = cfg["data"].get("categories", ["bottle"])
    summary_rows: List[Dict[str, Any]] = []
    for category in categories:
        try:
            metrics = run_category(cfg, category=category, output_root=experiment_dir, logger=logger)
            summary_rows.append({"category": category, **metrics})
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Category %s 실행 중 오류: %s", category, exc)

    summary_path = experiment_dir / "summary_metrics.csv"
    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    logger.info("완료. 결과는 %s 에 저장되었습니다.", experiment_dir)

    formatted_rows = _write_readable_summary(
        summary_rows, experiment_dir / "summary_metrics_pretty.csv"
    )
    _write_markdown_summary(
        formatted_rows,
        experiment_dir / "summary_metrics_pretty.md",
    )


def _format_metric_value(key: str, value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if math.isnan(numeric):
        return "N/A"

    if key in {"latency_ms_mean", "latency_ms_std"}:
        return f"{numeric:.3f} ms"
    if key == "vram_peak_bytes":
        gb_value = numeric / (1024 ** 3)
        return f"{gb_value:.3f} GB"
    if key == "fps":
        return f"{numeric:.2f}"
    if key.startswith("params_"):
        return f"{int(round(numeric)):,}"
    if key.endswith("_auroc") or key.endswith("_f1") or key == "pro_auroc":
        return f"{numeric:.4f}"
    return f"{numeric:.4f}"


def _write_readable_summary(
    rows: List[Dict[str, Any]], output_path: Path
) -> List[Dict[str, str]]:
    if not rows:
        return []

    formatted_rows: List[Dict[str, str]] = []
    for row in rows:
        formatted: Dict[str, str] = {}
        for key, value in row.items():
            if key == "category":
                formatted[key] = str(value)
            else:
                formatted[key] = _format_metric_value(key, value)
        formatted_rows.append(formatted)

    fieldnames = list(formatted_rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(formatted_rows)

    return formatted_rows


_METRIC_LABELS = {
    "category": "Category",
    "image_auroc": "Image AUROC",
    "image_f1": "Image F1",
    "pixel_f1": "Pixel F1",
    "region_f1": "Region F1",
    "pixel_auroc": "Pixel AUROC",
    "pro_auroc": "PRO AUROC",
    "latency_ms_mean": "Latency (ms)",
    "latency_ms_std": "Latency Std (ms)",
    "fps": "FPS",
    "params_total": "Params Total",
    "params_trainable": "Params Trainable",
    "flops": "FLOPs",
    "vram_peak_bytes": "VRAM Peak (GB)",
}


def _write_markdown_summary(
    rows: List[Dict[str, str]], output_path: Path
) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    headers = [_METRIC_LABELS.get(name, name.replace("_", " ").title()) for name in fieldnames]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        line = "| " + " | ".join(row[name] for name in fieldnames) + " |"
        lines.append(line)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = config_utils.load_config(args.config)
    if args.device:
        cfg.setdefault("model", {})["device"] = args.device
    run(cfg)


if __name__ == "__main__":
    main()

