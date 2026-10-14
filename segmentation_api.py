"""Local API for the educational mammography K-Means segmentation exercise."""
from __future__ import annotations

import base64
import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from skimage import color, io as skio, transform

from main import run_kmeans_segmentation, validate_kmeans_results

app = FastAPI(title="Mammography Image Lab API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

PALETTE = np.asarray([
    [31, 45, 66], [67, 171, 150], [211, 135, 155], [232, 188, 105]
], dtype=np.uint8)


def _data_url(array: np.ndarray, labels: bool = False) -> str:
    values = np.asarray(array)
    if labels:
        rgb = PALETTE[np.clip(values.astype(np.int32), 0, len(PALETTE) - 1)]
        image = Image.fromarray(rgb, "RGB")
    else:
        image = Image.fromarray((np.clip(values, 0, 1) * 255).round().astype(np.uint8), "L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Mammography Image Lab API"}


@app.post("/api/segment")
async def segment(file: UploadFile = File(...)):
    filename = file.filename or "mammogram"
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff")):
        raise HTTPException(status_code=415, detail="Upload a PNG, JPG, BMP, PGM or TIFF image.")
    try:
        raw = await file.read()
        image = skio.imread(io.BytesIO(raw))
        if image.ndim == 3:
            if image.shape[-1] == 4:
                image = image[..., :3]
            gray = color.rgb2gray(image)
        elif image.ndim == 2:
            gray = image.astype(np.float32)
            low, high = float(gray.min()), float(gray.max())
            if high <= low:
                raise ValueError("Image has no intensity variation.")
            gray = (gray - low) / (high - low)
        else:
            raise ValueError("Unsupported image dimensions.")
        gray = np.asarray(gray, dtype=np.float32)
        if not np.isfinite(gray).all():
            raise ValueError("Image contains non-finite pixel values.")

        original_shape = gray.shape
        # Limit interactive workload while preserving the broad intensity structure.
        if max(gray.shape) > 256:
            scale = 256 / max(gray.shape)
            shape = (max(2, int(gray.shape[0] * scale)), max(2, int(gray.shape[1] * scale)))
            gray = transform.resize(gray, shape, anti_aliasing=True).astype(np.float32)

        cluster_counts = (2, 3, 4)
        segmented = run_kmeans_segmentation(gray, cluster_counts)
        validation = validate_kmeans_results(gray, segmented, max_samples=10000)
        items = []
        for k in cluster_counts:
            values = segmented[k]
            validation_item = validation[k]
            items.append({
                "k": int(k),
                "preview": _data_url(values["segmented_image"], labels=True),
                "inertia": float(values["inertia"]),
                "silhouette": float(validation_item["silhouette"]),
                "cluster_sizes": {str(label): int(size) for label, size in validation_item["cluster_sizes"].items()},
                "centers": [float(v) for v in values["centers"]],
            })
        return {
            "filename": filename,
            "original": _data_url(gray),
            "width": int(gray.shape[1]),
            "height": int(gray.shape[0]),
            "original_width": int(original_shape[1]),
            "original_height": int(original_shape[0]),
            "downsampled": tuple(original_shape) != tuple(gray.shape),
            "results": items,
            "note": "K-Means groups grayscale pixel intensities only. These clusters are not validated anatomical structures and are not diagnostic.",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not process image: " + str(exc)) from exc
