import os
from pathlib import Path
from PIL import Image

from src.logger.logger import logging

INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
IMAGE_SIZE = 224

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
import os
from pathlib import Path
from PIL import Image

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
        logging.error(f"Error occurred during opening the image {path}: {e}")
        raise


def resize_image(image):
    try:
        resized_image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
        return resized_image
    except Exception as e:
        logging.error(f"Error Occurred during resizing the image: {e}")
        raise

def save_image(image, path):
    try : 
        image.save(path)

    except Exception as e:
        logging.error(f"Error occurred during saving the image {path}: {e}")
        raise

if __name__ == "__main__":
    try:
        if not INPUT_DIR.exists():
            raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

        for folder in os.listdir(INPUT_DIR):
            folder_path = os.path.join(INPUT_DIR, folder)
            

            if not os.path.isdir(folder_path):
                continue

            output_folder_path = os.path.join(OUTPUT_DIR, folder)
            os.makedirs(output_folder_path, exist_ok=True)
            
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                
                try:
                    img = open_image(image_path)
                    img = resize_image(img)
                    
                    save_path = os.path.join(output_folder_path, image_name)
                    save_image(img, save_path)
                    
                except Exception as e:
     
                    logging.error(f"Skipping image {image_name} due to error: {e}")
                    continue 

        logging.info("Image processing stage completed.")

    except Exception as e:
        logging.critical(f"Image processing pipeline failed: {e}")
        raise
