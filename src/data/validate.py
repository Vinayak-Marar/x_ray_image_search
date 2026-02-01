import os
import pandas as pd

ALLOWED_CATEGORIES = {"chest", "dental", "hand"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def validate_metadata(
    metadata_path: str = "data/metadata.csv",
    image_base_path: str = "data/raw",
):
    """
    Validates metadata and image files for X-ray image search pipeline.
    Raises ValueError if any validation fails.
    """

    # -------- 1. Load metadata --------
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    required_columns = {"image_name", "source_url", "category"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Metadata must contain columns: {required_columns}")

    # -------- 2. Duplicate image names --------
    if df["image_name"].duplicated().any():
        duplicates = df[df["image_name"].duplicated()]["image_name"].tolist()
        raise ValueError(f"Duplicate image names found: {duplicates}")

    # -------- 3. Category validation --------
    invalid_categories = set(df["category"]) - ALLOWED_CATEGORIES
    if invalid_categories:
        raise ValueError(f"Invalid categories found: {invalid_categories}")

    # -------- 4. File extension validation --------
    invalid_ext = [
        img for img in df["image_name"]
        if os.path.splitext(img)[1].lower() not in ALLOWED_EXTENSIONS
    ]
    if invalid_ext:
        raise ValueError(f"Invalid file extensions found: {invalid_ext}")

    # -------- 5. Metadata → Image files check --------
    missing_images = []
    for _, row in df.iterrows():
        img_path = os.path.join(image_base_path, row["category"], row["image_name"])
        if not os.path.exists(img_path):
            missing_images.append(img_path)

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} images listed in metadata are missing:\n"
            + "\n".join(missing_images)
        )

    # -------- 6. Image files → Metadata check --------
    all_image_files = set()
    for category in ALLOWED_CATEGORIES:
        category_dir = os.path.join(image_base_path, category)
        if not os.path.isdir(category_dir):
            continue

        for f in os.listdir(category_dir):
            if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS:
                all_image_files.add(f)

    metadata_images = set(df["image_name"])

    extra_images = all_image_files - metadata_images
    if extra_images:
        raise ValueError(
            f"{len(extra_images)} images exist without metadata:\n"
            + "\n".join(extra_images)
        )

    print("✅ Data validation passed successfully.")


if __name__ == "__main__":
    validate_metadata()
