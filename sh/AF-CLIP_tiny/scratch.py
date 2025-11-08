import torch

weights = torch.load("your_model.pt", map_location="cpu")

print(weights.keys())