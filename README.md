# X-Ray Image Search Interface

A semantic X-ray image retrieval system supporting both **image-to-image** and **text-to-image** search using CLIP embeddings.  
The system includes data validation, preprocessing, embedding generation, similarity search, and an interactive **Streamlit UI**.

> ⚠️ This project is intended for image retrieval only and does **not** perform medical diagnosis or clinical interpretation.

---

## Features

- 🔍 **Image-to-image search**: Retrieve structurally similar X-ray images  
- 🖼️ **Text-to-image search**: Retrieve X-ray images matching a natural language query  
- ✅ **Data validation** before processing using metadata (`metadata.csv`)  
- ⚙️ **Reproducible pipeline** with DVC  
- 🐳 **Containerized environment** using Docker  
- 📦 **Automated CI/CD pipeline** using GitHub Actions  
- 📊 **Interactive visualization** of search results via Streamlit  

---

## Dataset

- Approximately **500 public X-ray images**  
- Categories: Chest, Dental, Hand  
- Metadata CSV (`metadata.csv`) contains:
  - `image_name`
  - `source_url`
  - `category`  

### Dataset Validation

Performed **before preprocessing**:

- Metadata file presence and required columns  
- Duplicate image name detection  
- Category validation  
- Allowed image extensions: `.png`, `.jpg`, `.jpeg`  
- Consistency checks between metadata and image files:
  - Every metadata entry has a corresponding image file  
  - No extra images without metadata  

> ⚠️ Corrupted-image decoding is not performed; only logical consistency is verified.

---

## Preprocessing

- Images resized to match CLIP (ViT-B/32) input 

All data stages (raw → validated → processed) are tracked using **DVC** for reproducibility.

---

## Model

- **Model**: CLIP (ViT-B/32)  
- **Framework**: PyTorch  
- **Embedding vector length**: 512 for both image and text embeddings  
- **Training**: Pretrained, frozen (no fine-tuning)  
- **Encoders**: 
  - Image encoder → generates image embeddings  
  - Text encoder → generates text embeddings  

CLIP projects images and text into a **shared embedding space**, enabling multimodal retrieval.

---

## Similarity Search

- **Metric**: Cosine similarity  
- **Top-k results**: Default `k=5`, configurable `k=1–20`  
- **Minimum similarity thresholds**:  
  - Text-to-image: 0.25  
  - Image-to-image: 0.95  
- Only images above threshold are returned  
- Retrieval results are **displayed interactively via Streamlit**

---

## User Interface

- Built with **Streamlit**  
- Users can:
  - Upload an image query or enter a text query  
  - View top-k similar X-ray images in a **visual grid**  
- Allows quick and interactive assessment of retrieval relevance

---

## Project Structure

```text
x_ray_image_search/
│
├── data/
│   ├── raw/                # Raw X-ray images
│   ├── processed/          # Preprocessed images
│   └── metadata.csv
│
├── notebooks/
│   ├── data_validation.ipynb
│   ├── preprocessing.ipynb
│   ├── embedding_generation.ipynb
│   └── search.ipynb
│
├── src/
│   ├── validate_data.py
│   └── process.py
│
├── dvc.yaml
├── Dockerfile
├── ci.yaml
├── requirements.txt
└── README.md
````

---

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Vinayak-Marar/x_ray_image_search.git
cd x_ray_image_search
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull data using DVC

```bash
dvc pull
```

### 4. Run Streamlit UI

```bash
streamlit run src/app.py
```

* Upload a query image or type a text description
* View top-k retrieved images in an interactive grid

---

## Results

* Image-to-image search retrieves X-rays with similar structure
* Text-to-image search aligns descriptions with relevant images
* Minimum similarity thresholds improve retrieval quality and reduce irrelevant results

---

## Limitations

* Dataset is relatively small (~500 images)
* Corrupted-image detection is not implemented
* Retrieval uses brute-force cosine similarity (no ANN indexing)
* No medical interpretation or diagnosis

---

## Future Work

* Expand dataset to improve embedding diversity
* Integrate **FAISS** for faster retrieval
* Fine-tune CLIP on domain-specific medical images
* Support additional imaging modalities
* Add REST API endpoints or category-based filters

---

## Author

**Vinayak Marar**

```

---

