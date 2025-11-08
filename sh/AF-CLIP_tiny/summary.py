import clip

model, preprocess = clip.load(
        "ViT-B/16", 
        device="cpu", 
        jit=False, 
    )

print(model.visual)