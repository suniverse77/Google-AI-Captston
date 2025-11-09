import torch

path_tiny = "weight/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt"
path_af = ""

checkpoint = torch.load(path_tiny, map_location="cpu")

# 3. 실제 모델 가중치(state_dict)에 접근합니다.
model_weights = checkpoint['state_dict']

# 4. 이제 우리가 원했던 레이어 키 목록을 볼 수 있습니다!
print("Model Weight keys:", model_weights.keys())