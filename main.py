"""Batch Mammography Image Segmentation using K-Means Clustering on MIAS Dataset."""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn.cluster import KMeans


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """Loads a PGM/PNG mammogram and normalizes intensities to [0, 1]."""
    image = io.imread(image_path)
    if image.dtype == np.uint8:
        gray_image = image / 255.0
    elif image.dtype == np.uint16:
        gray_image = image / 65535.0
    else:
        gray_image = image.astype(float)
        gray_image = (gray_image - gray_image.min()) / (
            gray_image.max() - gray_image.min() + 1e-8
        )
    return gray_image


def run_kmeans_segmentation(gray_image: np.ndarray, n_clusters_list=None):
    """Applies K-Means clustering across specified cluster counts."""
    if n_clusters_list is None:
        n_clusters_list = [2, 3, 4]

    pixels = gray_image.reshape((-1, 1))
    results = {}

    for k in n_clusters_list:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        segmented = kmeans.labels_.reshape(gray_image.shape)
        results[k] = {
            "segmented_image": segmented,
            "centers": kmeans.cluster_centers_.flatten(),
        }
    return results


def save_batch_visualization(
    gray_image: np.ndarray, results: dict, filename: str, output_dir: str
):
    """Saves side-by-side segmentation visual grids for each scan."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 1. Original Scan
    axes[0, 0].imshow(gray_image, cmap="gray")
    axes[0, 0].set_title(f"Original Scan: {filename}")
    axes[0, 0].axis("off")

    # 2. Segmentations for k=2, 3, 4
    positions = [(0, 1), (1, 0), (1, 1)]
    for idx, (k, data) in enumerate(results.items()):
        row, col = positions[idx]
        axes[row, col].imshow(data["segmented_image"], cmap="viridis")
        axes[row, col].set_title(f"K-Means (k={k})")
        axes[row, col].axis("off")

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"seg_{filename}.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    data_dir = "data"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Search recursively inside data/ and all subfolders
    image_paths = (
        glob.glob(os.path.join(data_dir, "**", "*.pgm"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    )

    if not image_paths:
        print(
            f"[ERROR] No image files found in '{data_dir}/'. Extract your downloaded dataset into '{data_dir}/'!"
        )
        return

    print(
        f"[INFO] Found {len(image_paths)} mammogram scans. Starting batch segmentation..."
    )
    # Process all scans in the dataset
    for path in image_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        print(f"[PROCESSING] {filename}...")

        gray_img = load_and_preprocess_image(path)
        results = run_kmeans_segmentation(gray_img)
        save_batch_visualization(gray_img, results, filename, output_dir)

    print(
        f"[SUCCESS] Processed scans! Segmentation outputs saved in '{output_dir}/'."
    )
if __name__ == "__main__":
    main()