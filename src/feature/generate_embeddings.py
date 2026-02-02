import os
import numpy as np
from PIL import Image
import torch
import clip

from src.logger.logger import logging

MODEL_NAME = "ViT-B/32"
DEVICE = "cpu"
INPUT_DIR = "data/processed"
OUTPUT_DIR = "artifacts"

class EmbeddingGenerator:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        try:
            self.device = device
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            logging.info(f"CLIP model {model_name} loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise

    def compute_image_embedding(self, image_path):
        """
        Compute CLIP embedding for a single image.
        Returns a normalized numpy vector.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize

            logging.debug(f"Computed embedding for {image_path}")
            return embedding.cpu().numpy()[0]
        except Exception as e:
            logging.error(f"Error computing embedding for {image_path}: {e}")
            raise

    def generate_embeddings(self, input_dir=INPUT_DIR, save_dir=OUTPUT_DIR):
        """
        Generate embeddings for all images in input_dir and save to disk.
        """
        try:
            os.makedirs(save_dir, exist_ok=True)

            image_paths = []
            embeddings = []

            if not os.path.exists(input_dir):
                raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

            # Walk through all image files in input_dir
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        path = os.path.join(root, file)
                        try:
                            emb = self.compute_image_embedding(path)
                            embeddings.append(emb)
                            image_paths.append(path)
                        except Exception as e:
                            logging.warning(f"Skipping {path} due to error: {e}")
                            continue

            if not embeddings:
                logging.error("No embeddings were generated. Check your input directory.")
                return None, None

            embeddings = np.array(embeddings)
            image_paths = np.array(image_paths)

            # Save to disk
            np.save(os.path.join(save_dir, "image_embeddings.npy"), embeddings)
            np.save(os.path.join(save_dir, "image_paths.npy"), image_paths)

            logging.info(f"Saved {len(embeddings)} embeddings to {save_dir}")
            return embeddings, image_paths
            
        except Exception as e:
            logging.error(f"Embedding generation process failed: {e}")
            raise


if __name__ == "__main__":
    try:
        generator = EmbeddingGenerator()
        generator.generate_embeddings(INPUT_DIR, OUTPUT_DIR)
    except Exception as e:
        logging.critical(f"CLI execution failed: {e}")