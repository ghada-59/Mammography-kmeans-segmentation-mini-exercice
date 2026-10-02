# 🩻 Mammography Image Segmentation with K-Means

An academic image-processing exercise using mammography images to study unsupervised intensity clustering with **K-Means**.

## 🎯 Objective

The project demonstrates how grayscale pixel intensities can be grouped into clusters to create simple segmentation maps.

The implementation compares **k = 2, 3, and 4** for each image found under `data/`.

Importantly, the project now includes **quantitative internal validation** rather than relying only on visual inspection.

## ⚙️ Pipeline

1. Find supported mammography images recursively under `data/`.
2. Convert RGB/RGBA images to grayscale when necessary.
3. Normalize grayscale intensities to **[0, 1]**.
4. Flatten pixel intensities into a one-dimensional feature array.
5. Apply `sklearn.cluster.KMeans` with `k = 2, 3, 4`.
6. Reshape cluster labels into image-shaped segmentation maps.
7. Compute **K-Means inertia** for each k.
8. Compute the **silhouette score** for each k.
9. Save segmentation and validation visualizations.

For large images, the silhouette score uses a deterministic sample of at most 10,000 pixels to keep computation practical. The sample uses `random_state=42`.

K-Means uses `random_state=42` and `n_init=10` for reproducible initialization behavior.

## 📊 Validation

Two internal clustering measures are reported:

- **Inertia:** measures within-cluster squared distances. It generally decreases as k increases, so it should not be interpreted alone as proof that a larger k is better.
- **Silhouette score:** measures how well samples fit their own cluster compared with neighboring clusters. Higher values indicate better separation under this specific intensity-based representation.

These metrics help assess the clustering structure, but they **do not measure anatomical segmentation accuracy**.

No single k is hard-coded as clinically optimal. The values for k=2, 3, and 4 are provided for comparison.

## 🛠️ Technologies

Python · NumPy · scikit-image · scikit-learn · Matplotlib · K-Means

## 📁 Project structure

```
Mammography-kmeans-segmentation-mini-exercice/
├── data/                 # Mammography images supplied by the user
├── outputs/              # Generated segmentation and validation plots
├── main.py
├── requirements.txt
└── README.md
```

Generated outputs are excluded from version control.

## 🚀 Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the mammography images in `data/` (subfolders are supported), then run:

```bash
python main.py
```

The script prints inertia and silhouette values and saves the corresponding plots in `outputs/`.

## ⚠️ Scope and limitations

This is a **validated educational image-processing exercise**, not a clinical segmentation system.

- K-Means uses only pixel intensity and does not understand anatomy, lesions, or clinical context.
- Inertia and silhouette are **internal clustering metrics**, not segmentation ground-truth metrics.
- No annotated ground-truth masks are used, so Dice, IoU, sensitivity, and specificity cannot be claimed.
- Results depend on image characteristics, intensity normalization, and the tested values of k.
- A high silhouette score would indicate well-separated intensity clusters, not accurate lesion or tissue boundaries.
- The project does not implement lesion detection, diagnosis, or clinically validated computer-aided diagnosis.

The validation therefore supports the **technical quality of the clustering experiment**, while not establishing clinical validity.
