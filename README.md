# 🩻 Mammography Image Segmentation with K-Means

An academic image-processing exercise using mammography images to explore unsupervised intensity clustering with **K-Means**.

## 🎯 Objective

The project demonstrates how grayscale pixel intensities can be grouped into clusters to create simple image-segmentation maps.

The implementation compares **k = 2, 3, and 4** clusters for each image found under `data/`.

## ⚙️ Pipeline

1. Find supported mammography image files recursively under `data/`.
2. Convert RGB/RGBA images to grayscale when necessary.
3. Normalize grayscale intensities to the **[0, 1]** range.
4. Flatten pixel intensities into a one-dimensional feature array.
5. Apply `sklearn.cluster.KMeans` with `k = 2, 3, 4`.
6. Reshape the cluster labels into image-shaped segmentation maps.
7. Save a visualization containing the original image and the resulting segmentations.

K-Means uses a fixed `random_state=42` and `n_init=10` for reproducible initialization behavior.

## 🛠️ Technologies

Python · NumPy · scikit-image · scikit-learn · Matplotlib · K-Means

## 📁 Project structure

```
Mammography-kmeans-segmentation-mini-exercice/
├── data/                 # Place the mammography images here
├── outputs/              # Generated visualizations
├── main.py
├── requirements.txt
└── README.md
```

Generated images are excluded from version control.

## 🚀 Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Place the mammography images in `data/` (subfolders are supported), then run:

```bash
python main.py
```

The segmentation visualizations are written to `outputs/`.

## ⚠️ Scope and limitations

This is an **educational image-segmentation exercise** based only on pixel intensity clustering.

- K-Means clusters intensities; it does not understand anatomical structures or lesions.
- No annotated ground truth is used, so there is no quantitative Dice, IoU, sensitivity, or specificity evaluation.
- The choice of `k` is demonstrated for comparison and is not presented as a clinically optimal value.
- Results depend on image characteristics and preprocessing.
- The project does **not** implement lesion detection, diagnosis, or clinically validated computer-aided diagnosis.

This repository should therefore be interpreted as a learning/practice project rather than a clinical system.
