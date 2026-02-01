import os
import numpy as np
from PIL import Image
import torch
import clip
from tqdm import tqdm

MODEL_NAME = "ViT-B/32"
DEVICE = "cpu"
INPUT_DIR = "data/processed"
OUTPUT_DIR = "artifacts"

class EmbeddingGenerator:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def compute_image_embedding(self, image_path):
        """
        Compute CLIP embedding for a single image.
        Returns a normalized numpy vector.
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize

        return embedding.cpu().numpy()[0]

    def generate_embeddings(self, input_dir=INPUT_DIR, save_dir=OUTPUT_DIR):
        """
        Generate embeddings for all images in input_dir and save to disk.
        """
        os.makedirs(save_dir, exist_ok=True)

        image_paths = []
        embeddings = []

        # Walk through all image files in input_dir
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(root, file)
                    emb = self.compute_image_embedding(path)
                    embeddings.append(emb)
                    image_paths.append(path)

        embeddings = np.array(embeddings)
        image_paths = np.array(image_paths)

        # Save to disk
        np.save(os.path.join(save_dir, "image_embeddings.npy"), embeddings)
        np.save(os.path.join(save_dir, "image_paths.npy"), image_paths)

        print(f"[INFO] Saved {len(embeddings)} embeddings to {save_dir}")
        return embeddings, image_paths

# --------------------------
# Optional CLI usage
# --------------------------
if __name__ == "__main__":

    generator = EmbeddingGenerator()
    generator.generate_embeddings(INPUT_DIR, OUTPUT_DIR)
