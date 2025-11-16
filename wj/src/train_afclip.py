"""AF-CLIP 학습 스크립트."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_afclip_mvtec_dataloaders
from methods.afclip import AFClipMethod
from utils import config as config_utils
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.training_utils import focal_loss, l1_loss, patch_alignment_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AF-CLIP model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_afclip.yaml",
        help="학습 설정 YAML 경로 (프로젝트 루트 기준).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="강제로 사용할 디바이스 지정 (예: cpu, cuda:0)",
    )
    return parser.parse_args()


def train_category(
    cfg: Dict[str, Any],
    *,
    category: str,
    logger,
) -> None:
    """단일 카테고리에 대해 AF-CLIP 모델을 학습합니다."""
    logger.info("=== Category: %s ===", category)

    # 데이터셋 로드
    dataset_name = cfg["data"].get("dataset", "mvtec_afclip")
    if dataset_name != "mvtec_afclip":
        raise ValueError(f"현재는 mvtec_afclip만 지원합니다: {dataset_name}")

    dataloaders = build_afclip_mvtec_dataloaders(
        cfg["data"],
        category=category,
        transforms=None,
        mask_transforms=None,
    )
    
    # 학습용 데이터로더 생성 (shuffle=True)
    train_dataset = dataloaders["datasets"]["train"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"].get("batch_size", 8),
        shuffle=True,  # 학습 시 shuffle 필요
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=cfg["data"].get("pin_memory", False),
    )

    # 모델 설정
    model_cfg = dict(cfg["model"])
    device = cfg["model"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("runtime", {}).get("deterministic", True):
        torch.set_grad_enabled(True)  # 학습 시에는 grad 필요

    # 모델 생성 및 초기화
    model_wrapper = AFClipMethod(model_cfg, device=device)
    model_wrapper.setup()
    
    # 학습 가능한 파라미터만 활성화
    clip_model = model_wrapper._model
    assert clip_model is not None
    
    # 원본 코드와 동일하게 학습 가능한 파라미터만 활성화
    for param in clip_model.parameters():
        param.requires_grad_(False)
    
    # 학습 가능한 파라미터 활성화
    trainable_params = clip_model.get_trainable_parameters()
    for param in trainable_params:
        param.requires_grad_(True)

    # 옵티마이저 설정
    train_cfg = cfg.get("training", {})
    lr = float(train_cfg.get("lr", 0.0001))
    optimizer = torch.optim.Adam(trainable_params, lr=lr, betas=(0.5, 0.999))

    # Loss 가중치
    lambda1 = float(train_cfg.get("lambda1", 1.0))
    lambda2 = float(train_cfg.get("lambda2", 1.0))

    # 학습 루프
    epochs = int(train_cfg.get("epochs", 2))
    clip_model.train()

    for epoch in range(1, epochs + 1):
        total_loss = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            gts = batch["mask"].to(device)

            # Forward pass (학습 모드)
            predict_labels, predict_masks, img_tokens = clip_model.detect_forward_seg(
                imgs, args=model_wrapper._args
            )

            # Ground truth 마스크 크기 조정
            gts = F.interpolate(
                gts,
                size=predict_masks[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            gts[gts < 0.5] = 0
            gts[gts > 0.5] = 1

            # Loss 계산
            loss_cls = focal_loss(predict_labels, labels.float())
            loss_seg = focal_loss(predict_masks, gts) + l1_loss(predict_masks, gts)
            loss_patch = patch_alignment_loss(img_tokens, labels, gts)
            
            loss = loss_cls + lambda1 * loss_seg + lambda2 * loss_patch

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

        avg_loss = np.mean(total_loss)
        logger.info("Epoch: {}/{}, Loss: {:.6f}".format(epoch, epochs, avg_loss))

    # 학습된 가중치 저장
    save_dir = Path(cfg["training"].get("save_dir", "weights/afclip"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name_for_save = cfg["training"].get("dataset_name", "mvtec")
    prompt_path = save_dir / f"{dataset_name_for_save}_prompt.pt"
    adaptor_path = save_dir / f"{dataset_name_for_save}_adaptor.pt"
    
    torch.save(clip_model.state_prompt_embedding, prompt_path)
    torch.save(clip_model.adaptor.state_dict(), adaptor_path)
    
    logger.info("학습된 가중치 저장 완료:")
    logger.info("  - %s", prompt_path)
    logger.info("  - %s", adaptor_path)


def train(cfg: Dict[str, Any]) -> None:
    """AF-CLIP 모델 학습 메인 함수."""
    seed = cfg["experiment"].get("seed", 2025)
    set_seed(seed, deterministic=cfg.get("runtime", {}).get("deterministic", True))

    log_dir = Path(cfg["experiment"].get("output_root", "results/train_afclip"))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("train_afclip", log_dir=log_dir)
    logger.info("학습 시작. 로그 디렉토리: %s", log_dir)

    categories = cfg["data"].get("categories", ["bottle"])
    for category in categories:
        try:
            train_category(cfg, category=category, logger=logger)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Category %s 학습 중 오류: %s", category, exc)

    logger.info("학습 완료. 결과는 %s 에 저장되었습니다.", log_dir)


def main() -> None:
    args = parse_args()
    cfg = config_utils.load_config(args.config)
    if args.device:
        cfg.setdefault("model", {})["device"] = args.device
    train(cfg)


if __name__ == "__main__":
    main()

