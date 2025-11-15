import os
import argparse

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

from model.clip import CLIP
from model.tokenizer import tokenize

def load_model(args, device):
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    context_length = state_dict["positional_embedding"].shape[0]

    # projection 후 임베딩 차원 (공통 차원) → 현재: 512
    embed_dim = state_dict["text_projection"].shape[1]
    # ViT 임베딩 차원 → 현재: 256
    vision_embed_dim = state_dict["visual.conv1.weight"].shape[0]
    # 트랜스포머 임베딩 차원 → 현재: 256
    text_embed_dim = state_dict["ln_final.weight"].shape[0]

    # ViT layer 수 → 현재: 10
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    # 트랜스포머 layer 수 → 현재: 3
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # ViT 헤드 → 현재: 4
    vision_heads = vision_embed_dim // 64
    # 트랜스포머 헤드 → 현재: 4
    transformer_heads = text_embed_dim // 64

    # ViT 패치 크기 → 현재: 16
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    # ViT grid 크기 (한 변의 패치 개수) → 현재: 14
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    # 이미지 해상도 → 현재: 224
    image_resolution = vision_patch_size * grid_size

    model = CLIP(
        args=args,
        device=device,
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

    clip_transform = Compose([
        Resize(size=(args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        CenterCrop(size=(args.img_size, args.img_size)),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    return model, clip_transform

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, clip_transform = load_model(args=args, device=device)

    target_transform = transforms.Compose([
        transforms.Resize(size=clip_transform.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    
    # 원본 CLIP 모델의 모든 파라미터를 고정
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)
    clip_model = clip_model.to(device)

    clip_model.create_text_prompt(args=args, tokenizer=tokenize)

    train_dataset, test_dataset_dict = load_dataset(args, clip_transform, target_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch implemention of AF-CLIP')

    args = parser.parse_args()

    train(args)
