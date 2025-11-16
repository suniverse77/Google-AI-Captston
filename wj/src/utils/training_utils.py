"""Copied from WinCLIP (wj/src/WinClip/utils/training_utils.py)."""

import copy
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from utils.visualization import *


def get_optimizer_from_args(model, lr, weight_decay, **kwargs) -> torch.optim.Optimizer:
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)


def get_lr_schedule(optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dir_from_args(root_dir, class_name, **kwargs):
    exp_name = f"{kwargs['dataset']}-k-{kwargs['k_shot']}"

    csv_dir = os.path.join(root_dir, 'csv')
    csv_path = os.path.join(csv_dir, f"{exp_name}-indx-{kwargs['experiment_indx']}.csv")

    model_dir = os.path.join(root_dir, exp_name, 'models')
    img_dir = os.path.join(root_dir, exp_name, 'imgs')

    logger_dir = os.path.join(root_dir, exp_name, 'logger', class_name)

    log_file_name = os.path.join(
        logger_dir,
        f'log_{time.strftime("%Y-%m-%d-%H-%I-%S", time.localtime(time.time()))}.log',
    )

    model_name = f'{class_name}'

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    logger.start(log_file_name)

    logger.info(f"===> Root dir for this experiment: {logger_dir}")

    return model_dir, img_dir, logger_dir, model_name, csv_path


def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    """Focal loss for binary classification.
    
    Args:
        inputs: Predicted probabilities (B,) or (B, H, W)
        targets: Ground truth labels (B,) or (B, H, W)
        alpha: Weighting factor for positive/negative samples
        gamma: Focusing parameter
        reduction: 'mean' or 'sum'
    
    Returns:
        Loss value
    """
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def l1_loss(inputs, targets, reduction="mean"):
    """L1 loss wrapper."""
    return F.l1_loss(inputs, targets, reduction=reduction)


def patch_alignment_loss(img_tokens, labels, gts):
    """Patch alignment loss for AF-CLIP.
    
    Args:
        img_tokens: List of image token tensors from different layers
        labels: Image-level labels (B,)
        gts: Ground truth masks (B, H, W) or (B, 1, H, W)
    
    Returns:
        Loss value (torch.Tensor with gradient)
    """
    gts = gts.reshape(img_tokens[0].size(0), -1)
    labels = labels.reshape(labels.size(0), 1)
    new_gts = copy.copy(gts)
    
    # 정상 샘플만 있는 경우 (anomaly가 없는 경우) 0 반환
    if len(new_gts[new_gts == 0]) == 0:
        # Gradient를 유지하기 위해 첫 번째 img_token의 device와 dtype 사용
        return torch.tensor(0.0, device=img_tokens[0].device, dtype=img_tokens[0].dtype, requires_grad=True)
    
    new_gts[new_gts == 0] = -1
    b, l = new_gts.size()
    mask = torch.matmul(new_gts.reshape(b, l, 1), new_gts.reshape(b, 1, l))
    
    # Gradient를 유지하기 위해 첫 번째 img_token의 값으로 초기화
    total_sim = None
    for img_token in img_tokens:
        img_token = img_token[:, 1:, :]  # Remove CLS token
        img_token = F.normalize(img_token, dim=-1)
        sim = torch.matmul(img_token, img_token.permute(0, 2, 1))
        sim = sim[mask == -1].mean() - sim[mask == 1].mean()
        # ReLU와 유사하게 음수는 0으로
        sim = F.relu(sim)
        
        if total_sim is None:
            total_sim = sim
        else:
            total_sim = total_sim + sim
    
    return total_sim / len(img_tokens)


__all__ = [
    "get_optimizer_from_args",
    "get_lr_schedule",
    "setup_seed",
    "get_dir_from_args",
    "focal_loss",
    "l1_loss",
    "patch_alignment_loss",
]

