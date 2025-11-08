from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel, AutoProcessor

checkpoint = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"

model = CLIPModel.from_pretrained(checkpoint)

print(model.text_model)
print("==========================================")
print(model.vision_model)

print("\n=============== 비전 모델 설정 ===============")
print(model.config.vision_config)
