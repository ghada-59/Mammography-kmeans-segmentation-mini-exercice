# 🩺 Mammography Image Segmentation using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Domain](https://img.shields.io/badge/Domain-Biomedical%20Engineering-green.svg)
![License](https://img.shields.io/badge/License-MIT-orange.svg)

An unsupervised machine learning pipeline for segmenting mammographic images from the **MIAS (Mammographic Image Analysis Society) Dataset** using **K-Means Intensity Clustering** across $k \in \{2, 3, 4\}$ clusters.

---

## 📌 Project Overview

This project implements an automated, unsupervised machine learning pipeline for segmenting digital mammograms using **K-Means Intensity Clustering**. It serves a dual purpose: quantitative biomedical image analysis and reproducible pipeline engineering.

### 🔬 Scientific & Medical Purpose
In digital mammography, anatomical tissue density variations correspond directly to grayscale intensity distributions. K-Means clustering ($k \in \{2, 3, 4\}$) automatically partitions these structural regions without prior manual annotations:
* **Background Isolation ($k=2$):** Delineates the breast tissue parenchyma from non-tissue background air artifacts and acquisition noise.
* **Multi-Class Tissue Partitioning ($k=3, 4$):** Performs fine-grained intensity clustering to segment distinct radiological structures:
  1. **Adipose Tissue:** Low-density, dark parenchymal regions.
  2. **Fibroglandular Tissue:** Medium-density, intermediate parenchymal structures.
  3. **Radiopaque Lesions & Structures:** High-density, bright regions representing opacities, focal asymmetries, or potential microcalcifications.
* **Unsupervised CAD Preprocessing:** Evaluates non-parametric intensity clustering as an automated Region of Interest (ROI) extraction step for Computer-Aided Diagnosis (CAD) systems.

### 💻 Computational & Engineering Purpose
* **Automated Batch Processing:** Recursively scans, normalizes, and segments multi-format digital mammograms (MIAS `.pgm` dataset / `.png` / `.jpg`).
* **Feature Vectorization & Scalar Clustering:** Standardizes spatial 2D pixel matrices into normalized 1D intensity arrays ($[0, 1]$ range) to fit `sklearn.cluster.KMeans` efficiently.
* **Reproducible Output Serialization:** Automatically builds multi-panel visualization grids comparing original scans with segmentations across cluster counts, saving high-resolution outputs to a structured `outputs/` directory.

---

## 📂 Project Architecture

```text
mammography-kmeans-segmentation/
├── data/                           # MIAS dataset (.pgm files)
├── outputs/                        # Batch visual segmentation results
├── .gitignore                      # Git exclusion rules
├── main.py                         # Batch processing script
├── README.md                       # Documentation
└── requirements.txt                # Dependency specs