import os
from pathlib import Path
from PIL import Image
import numpy as np

from src.logger.logger import logging

INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
IMAGE_SIZE = 224

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def open_image(path):
    try :
        img = Image.open(path)
        logging.debug("Image opened successfully")
        return img
    
    except Exception as e:
        logging.error(f"Error occurred during opening the image: {e}")
        raise


def resize_image(image):
    try:
        resized_image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
        logging.debug("Image resized successfully")
        return resized_image
    except Exception as e:
        logging.error(f"Error Occurred during resizing the image: {e}")
        raise

def save_image(image, path):
    try : 
        image.save(path)
        logging.debug("Saved the image")

    except Exception as e:
        logging.error(f"Error occurred during saving the image: {e}")
        raise

if __name__ == "__main__":
    for folder in os.listdir(INPUT_DIR):
        output_dir = os.path.join(OUTPUT_DIR,folder)
        os.makedirs(output_dir, exist_ok=True)
        for image in os.listdir(os.path.join(INPUT_DIR,folder)):

            image_path = os.path.join(INPUT_DIR,folder,image)
            img = open_image(image_path)
            img = resize_image(img)
            
            save_path = os.path.join(OUTPUT_DIR,folder,image)
            save_image(img, save_path)

