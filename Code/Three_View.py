# Three_View.py
# Multi-view generation: PCA, UMAP, PHATE
import numpy as np
from sklearn.decomposition import PCA
import umap
import phate


def generate_pca_view(X, n_components=20, seed=42):
    max_comp = min(X.shape[0], X.shape[1])
    safe_comp = min(int(n_components), max_comp) if max_comp >= 1 else 1

    pca = PCA(n_components=safe_comp, random_state=seed)
    X_pca = pca.fit_transform(X)
    return X_pca


def generate_umap_view(X, n_components=10, seed=42):
    max_comp = min(X.shape[0], X.shape[1])
    safe_comp = min(int(n_components), max_comp) if max_comp >= 1 else 1

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, n_components=safe_comp, random_state=seed)
    X_umap = reducer.fit_transform(X)
    return X_umap


def generate_phate_view(X, n_components=10, seed=42):
    max_comp = min(X.shape[0], X.shape[1])
    safe_comp = min(int(n_components), max_comp) if max_comp >= 1 else 1

    phate_op = phate.PHATE(n_components=safe_comp, knn=10, t='auto', random_state=seed, verbose=False)
    X_phate = phate_op.fit_transform(X)
    return X_phate
