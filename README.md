# Dimensionality Reduction and Clustering Toolbox

This toolbox provides Python implementations of popular dimensionality reduction techniques (ACP, t-SNE, UMAP) and a clustering algorithm (KMeans-birch).

## Table of Contents

1. [Introduction](#Introduction)
2. [Dependencies](#Dependencies)
3. [Examples](#Examples)


## Introduction

This repository contains Python scripts implementing the following algorithms:

- **ACP (Principal Component Analysis):** `main/experiments/template_branche_mazlin.py`
- **t-SNE (t-distributed Stochastic Neighbor Embedding):** `main/experiments/template_branche_hatim.py`
- **UMAP (Uniform Manifold Approximation and Projection):** `main/experiments/template_branche_abdelhak.py`

These algorithms are commonly used for dimensionality reduction and clustering in machine learning and data analysis.

## Dependencies

Make sure you have the following dependencies installed:

- `numpy`
- `scikit-learn`
- `umap-learn`
- `matplotlib`
- `sentence_tranformers`
- `joblib`

You can install them using:

```bash
pip install numpy scikit-learn umap-learn matplotlib sentence_tranformers joblib

## Visualization results
### PCA Visualization with Kmeans Clusters

![PCa - kmeans](https://github.com/Ahak99/Exam---Data-Engineering/assets/101395769/4818d111-6288-4795-99a4-539e5f93b0f0)

### PCA Visualization with Birch Clusters

![pca -birch](https://github.com/Ahak99/Exam---Data-Engineering/assets/101395769/d161c1c5-cbb7-4724-ba15-b723f10c688e)

### TSNE Visualization with Kmeans Clusters

![tsne - kmeans](https://github.com/Ahak99/Exam---Data-Engineering/assets/101395769/e3b1a886-5093-4f63-92cf-aa8bd86d42c4)

### TSNE Visualization with Birch Clusters

![TSNE - BIRCH](https://github.com/Ahak99/Exam---Data-Engineering/assets/101395769/4b32fa30-c397-400f-a9d5-b212afa95cdf)

### UMAP Visualization with KMeans Clusters

![umap - kmeans](https://github.com/Ahak99/Exam---Data-Engineering/assets/101395769/0608c7e9-cb86-4d62-a8ac-194315a889db)

### UMAP Visualization with Birch Clusters

![umap - birch](https://github.com/Ahak99/Exam---Data-Engineering/assets/101395769/174ef21c-3121-4f9e-8583-3ecbe9e51f17)
