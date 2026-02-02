import streamlit as st
import numpy as np
import os
from src.utils.utils import load_clip_model, search_by_text, search_by_image
from src.logger.logger import logging

# Configuration
EMBEDDINGS_PATH = "artifacts/image_embeddings.npy"
PATHS_PATH = "artifacts/image_paths.npy"
MODEL_NAME = "ViT-B/32"
DEVICE = "cpu"
DEFAULT_TOP_K = 5

@st.cache_resource
def load_model():
    try:
        logging.info(f"Loading CLIP model: {MODEL_NAME}")
        return load_clip_model(MODEL_NAME, DEVICE)
    except Exception as e:
        logging.error(f"Failed to load model in Streamlit: {e}")
        st.error("Could not load the AI model. Please check the logs.")
        return None, None

model, preprocess = load_model()

@st.cache_data
def load_embeddings():
    try:
        if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(PATHS_PATH):
            logging.error("Embedding or Path files not found in artifacts/")
            raise FileNotFoundError("DVC artifacts missing. Run 'dvc pull'?")

        embeddings = np.load(EMBEDDINGS_PATH)
        image_paths = np.load(PATHS_PATH)
        
        # Logic for category extraction
        categories = [p.split("\\")[-2] for p in image_paths]
        logging.info(f"Successfully loaded {len(embeddings)} embeddings")
        return embeddings, image_paths, categories
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        st.error("Data files are missing or corrupted.")
        return np.array([]), np.array([]), []

embeddings, image_paths, image_categories = load_embeddings()

# Normalize paths for cross-platform compatibility
image_paths = [p.replace("\\", "/") for p in image_paths]

# --- Streamlit UI ---

st.title("X-ray Image Search 🔍")
st.write("Search X-ray images by **text** or **image** using CLIP embeddings.")

mode = st.radio("Search mode", ["Text", "Image"])

# Top-K input
top_k = int(st.number_input("Number of results to show", min_value=1, max_value=20, value=DEFAULT_TOP_K, step=1))

# Category filter
if len(image_categories) > 0:
    unique_categories = sorted(list(set(image_categories)))
    selected_category = st.selectbox("Filter by category", ["All"] + unique_categories)

    # Filter embeddings by category
    if selected_category != "All":
        filtered_indices = [i for i, cat in enumerate(image_categories) if cat == selected_category]
        filtered_embeddings = embeddings[filtered_indices]
        filtered_paths = [image_paths[i] for i in filtered_indices]
    else:
        filtered_embeddings = embeddings
        filtered_paths = image_paths
else:
    st.warning("No categories found. Is the metadata correct?")
    filtered_embeddings = embeddings
    filtered_paths = image_paths

# --- Text search ---

if mode == "Text":
    query = st.text_input("Enter your text query")
    if st.button("Search") and query:
        try:
            results = search_by_text(query, model, filtered_embeddings, filtered_paths, DEVICE, top_k, 0.25)
            if not results:
                st.info("No relevant images found for this query.")
            else:
                st.subheader("Top Results:")
                for i, (path, score) in enumerate(results, 1):
                    cols = st.columns([1, 3, 1])
                    cols[0].write(f"**Rank {i}**")
                    if os.path.exists(path):
                        cols[1].image(path, width="stretch")
                    else:
                        cols[1].warning(f"File missing: {path}")
                    cols[2].write(f"Score: **{score:.3f}**")
        except Exception as e:
            logging.error(f"Text search UI error: {e}")
            st.error("An error occurred during text search.")

# --- Image search ---

if mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file and st.button("Search"):
        try:
            results = search_by_image(uploaded_file, model, preprocess, filtered_embeddings, filtered_paths, DEVICE, top_k, 0.90)
            if not results:
                st.info("No relevant images found for this query.")
            else:
                st.subheader("Top Results:")
                for i, (path, score) in enumerate(results, 1):
                    cols = st.columns([1, 3, 1])
                    cols[0].write(f"**Rank {i}**")
                    if os.path.exists(path):
                        cols[1].image(path, width="stretch")
                    else:
                        cols[1].warning(f"File missing: {path}")
                    cols[2].write(f"Score: **{score:.3f}**")
        except Exception as e:
            logging.error(f"Image search UI error: {e}")
            st.error("An error occurred during image search.")