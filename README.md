# 🩻 Mammography Image Segmentation with K-Means

An academic image-processing exercise using the **MIAS dataset** to explore unsupervised intensity clustering with **K-Means**.

## 🎯 Objective

The project studies how grayscale intensity values can be grouped into clusters to produce simple image-segmentation masks.

The current implementation evaluates **k = 2, 3, and 4 clusters**.

## ⚙️ Pipeline

1. Load mammography images from the dataset.
2. Convert image intensities to floating-point values in the **[0, 1]** range.
3. Flatten pixel intensities into a one-dimensional feature array.
4. Apply `sklearn.cluster.KMeans` with `k = 2, 3, 4`.
5. Reshape cluster labels into segmentation images.
6. Save side-by-side visualizations for comparison.

## 🛠️ Technologies

Python · NumPy · scikit-image · scikit-learn · Matplotlib · K-Means

## ⚠️ Scope and limitations

This is an **educational image-segmentation exercise** based only on intensity clustering.

It does **not** implement lesion detection, diagnosis, clinical CAD, or clinically validated tissue segmentation. The project also does not use annotated ground truth to quantitatively evaluate segmentation quality.

## 🚀 Run

```bash
pip install -r requirements.txt
python main.py
```
