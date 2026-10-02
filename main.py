"""Batch mammography image segmentation using K-Means clustering."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io
from sklearn.cluster import KMeans


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CLUSTERS = (2, 3, 4)
VALID_EXTENSIONS = {".pgm", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_and_preprocess_image(image_path):
    """Load an image and normalize grayscale intensities to [0, 1]."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = io.imread(image_path)

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[..., :3]
        if image.shape[2] != 3:
            raise ValueError(f"Unsupported channel count: {image.shape[2]}")
        gray_image = color.rgb2gray(image)
    elif image.ndim == 2:
        gray_image = image.astype(np.float32)
        min_value = float(gray_image.min())
        max_value = float(gray_image.max())
        if max_value <= min_value:
            raise ValueError(f"Image has no intensity variation: {image_path}")
        gray_image = (gray_image - min_value) / (max_value - min_value)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.asarray(gray_image, dtype=np.float32)


def run_kmeans_segmentation(gray_image, n_clusters_list=None):
    """Apply K-Means to pixel intensities for the requested cluster counts."""
    gray_image = np.asarray(gray_image)
    if gray_image.ndim != 2:
        raise ValueError("gray_image must be a 2D grayscale image.")
    if not np.isfinite(gray_image).all():
        raise ValueError("gray_image contains NaN or infinite values.")

    clusters = tuple(DEFAULT_CLUSTERS if n_clusters_list is None else n_clusters_list)
    if not clusters:
        raise ValueError("At least one cluster count is required.")
    if any(not isinstance(k, (int, np.integer)) or k < 2 for k in clusters):
        raise ValueError("Each cluster count must be an integer >= 2.")
    if len(set(clusters)) != len(clusters):
        raise ValueError("Cluster counts must be unique.")

    pixels = gray_image.reshape(-1, 1)
    if max(clusters) > len(pixels):
        raise ValueError("The largest cluster count cannot exceed the number of pixels.")

    results = {}
    for k in clusters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        results[k] = {
            "segmented_image": kmeans.labels_.reshape(gray_image.shape),
            "centers": kmeans.cluster_centers_.ravel(),
            "inertia": float(kmeans.inertia_),
        }

    return results


def save_batch_visualization(gray_image, results, filename, output_dir):
    """Save one visualization containing the original image and segmentations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = list(results.items())
    figure, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    axes[0].imshow(gray_image, cmap="gray")
    axes[0].set_title(f"Original Scan: {filename}")
    axes[0].axis("off")

    for axis, (k, data) in zip(axes[1:], items):
        axis.imshow(data["segmented_image"], cmap="viridis")
        axis.set_title(f"K-Means (k={k})")
        axis.axis("off")

    for axis in axes[1 + len(items):]:
        axis.axis("off")

    figure.tight_layout()
    output_path = output_dir / f"seg_{filename}.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def find_image_paths(data_dir):
    """Find supported image files recursively in a deterministic order."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    return sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = find_image_paths(DATA_DIR)

    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files found in '{DATA_DIR}'."
        )

    print(f"[INFO] Found {len(image_paths)} mammogram scans.")
    for path in image_paths:
        filename = path.stem
        print(f"[PROCESSING] {filename}...")
        gray_image = load_and_preprocess_image(path)
        results = run_kmeans_segmentation(gray_image)
        save_batch_visualization(
            gray_image, results, filename, OUTPUT_DIR
        )

    print(f"[SUCCESS] Outputs saved in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
