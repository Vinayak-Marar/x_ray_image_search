import numpy as np
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Union, Optional

from src.logger.logger import logging

def load_clip_model(model_name: str = "ViT-B/32", device: Optional[str] = None):
    """
    Loads CLIP model with error handling and device management.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        logging.info(f"Successfully loaded {model_name} on {device}")
        return model, preprocess
    except Exception as e:
        logging.error(f"Failed to load CLIP model: {e}")
        raise RuntimeError(f"Could not initialize CLIP: {e}")

def normalize(v: np.ndarray) -> np.ndarray:
    """
    Safely normalizes a vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def search_by_text(query: str, model, embeddings: np.ndarray, image_paths: np.ndarray, device: str = "cpu", top_k: int = 5, min_similarity: float = 0.25) -> List[Tuple[str, float]]:
    """
    Text-to-Image search with validation and optimized similarity calculation.
    """
    try:
        text = clip.tokenize([query]).to(device)
        
        with torch.no_grad():
            text_embedding = model.encode_text(text)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        
        text_embedding = text_embedding.cpu().numpy()
        
        sims = cosine_similarity(text_embedding, embeddings)[0]
        
        top_indices = sims.argsort()[::-1][:top_k]
        
        results = [
            (str(image_paths[i]), float(sims[i])) 
            for i in top_indices 
            if sims[i] >= min_similarity
        ]
        
        return results
    except Exception as e:
        logging.error(f"Text search failed for query '{query}': {e}")
        return []

def search_by_image(uploaded_image: Union[str, Image.Image], model, preprocess, embeddings: np.ndarray, image_paths: np.ndarray, device: str = "cpu", top_k: int = 5, min_similarity: float = 0.95) -> List[Tuple[str, float]]:
    """
    Image-to-Image search with input validation and memory safety.
    """
    try:
        image = Image.open(uploaded_image).convert("RGB")

        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_embedding = model.encode_image(image_tensor)
            img_embedding /= img_embedding.norm(dim=-1, keepdim=True)
            
        img_embedding = img_embedding.cpu().numpy()
        sims = cosine_similarity(img_embedding, embeddings)[0]
        
        top_indices = sims.argsort()[::-1][:top_k]
        
        results = [
            (str(image_paths[i]), float(sims[i])) 
            for i in top_indices 
            if sims[i] >= min_similarity
        ]
        
        return results
    except Exception as e:
        logging.error(f"Image search failed: {e}")
        return []