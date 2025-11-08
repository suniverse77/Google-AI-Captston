import torch

path_tiny = "AF-CLIP_tiny/weight/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt"
path_clip = "AF-CLIP/download/clip/ViT-L-14-336px.pt"

ckpt_tiny = torch.load(path_tiny, map_location="cpu")
ckpt_clip = torch.load(path_clip, map_location="cpu", weights_only=False)

model_weights = ckpt_tiny['state_dict']

print(model_weights.keys())
print("========================================")
print(ckpt_clip.state_dict().keys())

print(model_weights["_image_encoder.module.visual.class_embedding"].shape)
