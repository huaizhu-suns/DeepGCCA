import numpy as np


def pretreat_data(X):
    """
    Preprocess a single-cell gene expression matrix.

    Steps:
    1. Filter cells with < 200 expressed genes.
    2. Normalize each cell to total count = 10^4.
    3. Apply ln(X + 1) transformation.
    4. Select top 2000 highly variable genes (if > 2000 genes).
    5. Z-score standardize and remove constant genes.

    Args:
        X: np.ndarray of shape (n_cells, n_genes).

    Returns:
        X_pre: np.ndarray, preprocessed matrix.
    """
    mask = np.sum(X > 0, axis=1) >= 200
    X = X[mask]

    X = X / (np.sum(X, axis=1, keepdims=True) + 1e-6) * 1e4
    X = np.log(X + 1)

    if X.shape[1] > 2000:
        var = np.var(X, axis=0)
        top_idx = np.argsort(var)[-2000:]
        X = X[:, top_idx]

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    keep = sigma > 0
    X = ((X - mu) / sigma)[:, keep]

    return X
