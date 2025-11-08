import torch

path = "weight/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt"

checkpoint = torch.load(path, map_location="cpu")

# 2. checkpoint의 최상위 키를 확인합니다. (이게 바로 'state_dict'가 나온 부분입니다)
print("Checkpoint keys:", checkpoint.keys())

# 3. 실제 모델 가중치(state_dict)에 접근합니다.
model_weights = checkpoint['state_dict']

# 4. 이제 우리가 원했던 레이어 키 목록을 볼 수 있습니다!
print("Model Weight keys:", model_weights.keys())