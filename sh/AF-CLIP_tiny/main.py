import os
import random
import logging
import argparse
from tqdm import tqdm
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from dataset import *
from util.utils import eval_all_class
from util.loss_fn import focal_loss, l1_loss, patch_alignment_loss
from clip.clip import load, tokenize, available_models

from model.clip import CLIP
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def setup_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def load_model(path, device):
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

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        # ResNet 로직 (TinyCLIP은 ViT이므로 이 부분은 실행되지 않음)
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    model.load_checkpoint(state_dict)

    if str(device) == "cpu":
        model.float()

    return model.eval(), _transform(model.visual.input_resolution)
    

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # log 저장
    logger = get_logger(os.path.join(args.log_dir, '{}_{}_s{}.txt'.format(args.dataset, args.fewshot, args.seed)))
    print_args(logger, args)

    clip_model, clip_transform = load_model(path=args.model_weight, device=device)

    clip_transform.transforms[0] = transforms.Resize(size=(args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC)
    clip_transform.transforms[1] = transforms.CenterCrop(size=(args.img_size, args.img_size))
    target_transform = transforms.Compose([
        transforms.Resize(size=clip_transform.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    clip_model.eval()
    
    # CLIP 모델의 모든 파라미터를 고정
    for param in clip_model.parameters():
        param.requires_grad_(False)
    
    clip_model = clip_model.to(device)
    clip_model.insert(args=args, tokenizer=tokenize, device=device)

    test_dataset_mvtec = MVTecDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_visa = VisaDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
    
    all_test_dataset_dict = {
        "mvtec": test_dataset_mvtec,
        "visa": test_dataset_visa,
    }

    if len(args.test_dataset) < 1:
        test_dataset_dict = all_test_dataset_dict
    else:
        test_dataset_dict = {}
        for ds_name in args.test_dataset:
            test_dataset_dict[ds_name] = all_test_dataset_dict[ds_name]
    if args.dataset in test_dataset_dict:
        del test_dataset_dict[args.dataset]

    #     
    if args.dataset == 'mvtec':
        train_dataset = test_dataset_mvtec
    else:
        train_dataset = test_dataset_visa
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 가중치 존재하는 경우
    if args.weight is not None:
        clip_model.state_prompt_embedding = torch.load(os.path.join(args.weight, "{}_prompt.pt".format(args.dataset)), map_location=torch.device('cpu'), weights_only=False)
        clip_model.adaptor = torch.load(os.path.join(args.weight, "{}_adaptor.pt".format(args.dataset)), map_location=torch.device('cpu'), weights_only=False)
    # 가중치 없는 경우 → 어댑터, 프롬프트 파인튜닝
    else:
        optimizer = torch.optim.Adam(clip_model.get_trainable_parameters(), lr=args.lr, betas=(0.5, 0.999))
       
        for epoch in range(1, args.epochs + 1):
            total_loss = []
            
            for items in tqdm(train_dataloader):
                imgs, labels, gts = items[:3]
                labels = labels.to(device)
                imgs = imgs.to(device)
                gts = gts.to(device)
                predict_labels, predict_masks, img_tokens = clip_model.detect_forward_seg(imgs, args=args)
                gts = F.interpolate(gts, size=predict_masks[0].shape[-2:], mode='bilinear')
                gts[gts < 0.5] = 0
                gts[gts > 0.5] = 1
                
                loss = focal_loss(predict_labels, labels) + args.lambda1 * (focal_loss(predict_masks, gts) + l1_loss(predict_masks, gts)) + args.lambda2 * patch_alignment_loss(img_tokens, labels, gts) 
                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
                
            logger.info("Epoch: {}/{}, Loss: {:.6f}".format(epoch, args.epochs, np.mean(total_loss)))

    for dataset_name, test_ds in test_dataset_dict.items():
        logger.info("---------------------------{}------------------------------".format(dataset_name))
        eval_all_class(clip_model, dataset_name, test_ds, args, logger, device)
        logger.info("-------------------------------------------------------------")

      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch implemention of AF-CLIP')
    
    parser.add_argument('--clip_download_dir', type=str, default='./download/clip/', help='training dataset')
    parser.add_argument('--data_dir', type=str, default='./data', help='training dataset')
    parser.add_argument('--dataset', type=str, default='mvtec', help='training dataset', choices=['mvtec', 'visa'])
    parser.add_argument('--model', type=str, default="ViT-L/14@336px", help='model')
    parser.add_argument('--model_weight', type=str, default="./weight/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt", help='model')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='label combination')
    parser.add_argument('--epochs', type=int, default=2, help='training epoch')
    parser.add_argument('--prompt_len', type=int, default=12, help='prompt length')
    parser.add_argument('--category', type=str, default=None, help='normal class')
    parser.add_argument('--fewshot', type=int, default=0, help='few shot num')
    parser.add_argument('--seed', type=int, default=122, help='seed')
    parser.add_argument('--log_dir', type=str, default='./log/', help='log dir')
    parser.add_argument('--suffix', type=str, default='defect', help='prompt suffix')
    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--feature_layers', nargs='+', type=int, default=[6, 12, 18, 24], help='choose vit layers to extract features')
    parser.add_argument('--test_dataset', nargs='+', type=str, default=[], help='choose vit layers to extract features')
    parser.add_argument('--weight', type=str, default=None, help='load weight path')
    parser.add_argument('--vis', type=int, default=0, help='visualization results')
    parser.add_argument('--vis_dir', type=str, default='./vis_results/', help='visualization results dir')
    parser.add_argument('--memory_layers',  nargs='+', type=int, default=[6, 12, 18, 24], help='choose resnet layers to store and compare features')
    parser.add_argument('--lambda1', type=float, default=1, help='lambda1 for loss')
    parser.add_argument('--lambda2', type=float, default=1, help='lambda2 for loss')
    
    args = parser.parse_args()
    
    args.seed = setup_seed(args.seed)
    train(args)
    