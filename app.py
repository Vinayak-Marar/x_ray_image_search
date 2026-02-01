import streamlit as st
import numpy as np
from src.utils.utils import load_clip_model, search_by_text, search_by_image

EMBEDDINGS_PATH = "artifacts/image_embeddings.npy"
PATHS_PATH = "artifacts/image_paths.npy"
MODEL_NAME = "ViT-B/32"
DEVICE = "cpu"
DEFAULT_TOP_K = 5

@st.cache_resource
def load_model():
    return load_clip_model(MODEL_NAME, DEVICE)

model, preprocess = load_model()

@st.cache_data
def load_embeddings():
    embeddings = np.load(EMBEDDINGS_PATH)
    image_paths = np.load(PATHS_PATH)
    
    categories = [p.split("\\")[-2] for p in image_paths]
    return embeddings, image_paths, categories

embeddings, image_paths, image_categories = load_embeddings()


# Streamlit UI

st.title("X-ray Image Search 🔍")
st.write("Search X-ray images by **text** or **image** using CLIP embeddings.")

mode = st.radio("Search mode", ["Text", "Image"])

# Top-K input
top_k = int(st.number_input("Number of results to show", min_value=1, max_value=20, value=DEFAULT_TOP_K, step=1))

# Category filter
unique_categories = sorted(list(set(image_categories)))
selected_category = st.selectbox("Filter by category", ["All"] + unique_categories)


# Filter embeddings by category

if selected_category != "All":
    filtered_indices = [i for i, cat in enumerate(image_categories) if cat == selected_category]
    filtered_embeddings = embeddings[filtered_indices]
    filtered_paths = image_paths[filtered_indices]
else:
    filtered_embeddings = embeddings
    filtered_paths = image_paths

# Text search

if mode == "Text":
    query = st.text_input("Enter your text query")
    if st.button("Search") and query:
        results = search_by_text(query, model, filtered_embeddings, filtered_paths, DEVICE, top_k)
        if len(results) == 0:
            st.info("No relevant images found for this query.")

        else:

            st.subheader("Top Results:")

            # Display results in table
            cols = st.columns([1, 3, 1])
            cols[0].markdown("**Rank**")
            cols[1].markdown("**Image**")
            cols[2].markdown("**Similarity Score**")

            for i, (path, score) in enumerate(results, 1):
                cols = st.columns([1, 3, 1])
                cols[0].write(i)
                cols[1].image(path, width=500)
                cols[2].write(f"{score:.3f}")

# Image search

if mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file and st.button("Search"):
        results = search_by_image(uploaded_file, model, preprocess, filtered_embeddings, filtered_paths, DEVICE, top_k)
        if len(results) == 0:
            st.info("No relevant images found for this query.")
        
        else:
            st.subheader("Top Results:")

            # Display results in table
            cols = st.columns([1, 3, 1])
            cols[0].markdown("**Rank**")
            cols[1].markdown("**Image**")
            cols[2].markdown("**Similarity**")

            for i, (path, score) in enumerate(results, 1):
                cols = st.columns([1, 3, 1])
                cols[0].write(i)
                cols[1].image(path, width=500)
                cols[2].write(f"{score:.3f}")
