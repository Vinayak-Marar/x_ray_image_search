# src/util/helpers.py
import numpy as np
from PIL import Image
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity

def load_clip_model(model_name="ViT-B/32", device="cpu"):
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess

def normalize(v):
    return v / np.linalg.norm(v)

def search_by_text(query, model, embeddings, image_paths, device="cpu", top_k=5, min_similarity=0.25):
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding.cpu().numpy()
    sims = cosine_similarity(text_embedding, embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [(image_paths[i], sims[i]) for i in top_indices if sims[i]>=min_similarity]

def search_by_image(uploaded_image, model, preprocess, embeddings, image_paths, device="cpu", top_k=5, min_similarity=0.95):
    image = Image.open(uploaded_image).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_embedding = model.encode_image(image_tensor)
        img_embedding /= img_embedding.norm(dim=-1, keepdim=True)
    img_embedding = img_embedding.cpu().numpy()
    sims = cosine_similarity(img_embedding, embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [(image_paths[i], sims[i]) for i in top_indices if sims[i] >= min_similarity]
