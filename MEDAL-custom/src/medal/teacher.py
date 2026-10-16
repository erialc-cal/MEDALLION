"""
Teacher embedding computation and hyperparameter grid construction.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from medal._paths import TEACHER_PARAM_KEY


# ------------------------------------------------------------------
# Default sweep ranges (log-spaced grid of n_grid points)
# ------------------------------------------------------------------

_DEFAULT_RANGES = {
    "umap":     (5, 500),
    "tsne":     (5, 500),
    "spectral": (5, 200),
    "phate":    (5, 500),
    "isomap":   (5, 200),
}

_DEFAULT_N_GRID = 15


def build_param_grid(
    teacher: str,
    n_components: int = 2,
    n_grid: int = _DEFAULT_N_GRID,
    param_range: Optional[tuple] = None,
    custom_grid: Optional[list] = None,
) -> list[dict]:
    """
    Build a list of teacher hyperparameter configs for a sweep.

    Parameters
    ----------
    teacher : str
        One of "umap", "tsne", "spectral", "phate", "isomap", "pca".
    n_components : int
        Embedding dimensionality (default 2).
    n_grid : int
        Number of log-spaced grid points for the primary hyperparameter.
    param_range : (min, max), optional
        Override the default range for the primary hyperparameter.
    custom_grid : list of dict, optional
        Skip the default grid entirely and return this list as-is.

    Returns
    -------
    list of dict
        Each dict is a valid tc config to pass to
        get_teacher_embeddings or medal.sweep.run_teacher_sweep.
    """
    if custom_grid is not None:
        return custom_grid

    teacher = teacher.lower()

    if teacher == "pca":
        return [{"n_components": n_components}]

    if teacher not in _DEFAULT_RANGES:
        raise ValueError(
            f"Unknown teacher {teacher!r}. "
            f"Known: {list(_DEFAULT_RANGES.keys()) + ['pca']}"
        )

    lo, hi = param_range or _DEFAULT_RANGES[teacher]
    values = np.unique(np.logspace(np.log10(lo), np.log10(hi), n_grid).astype(int)).tolist()
    param_key = TEACHER_PARAM_KEY[teacher]

    base = {"n_components": n_components}
    if teacher == "umap":
        base["min_dist"] = 0.1
        return [{**base, param_key: v} for v in values]
    return [{**base, param_key: v} for v in values]


def get_teacher_embeddings(
    method: str,
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 0,
    save_path: Optional[str | Path] = None,
    **teacher_kwargs,
) -> np.ndarray:
    """
    Compute a teacher embedding for array *X*.

    Parameters
    ----------
    method : str
        One of "umap", "tsne", "pca", "isomap", "spectral", "phate".
    X : array-like of shape (n_samples, n_features)
        Input data.
    n_components : int
        Embedding dimensionality.
    random_state : int
        Random seed for reproducible teacher embeddings (default 0).
        Forwarded to teachers that support it; ignored for isomap/phate.
    save_path : str or Path, optional
        If given, pickle the fitted teacher model to this path.
    **teacher_kwargs
        Extra keyword arguments forwarded to the teacher algorithm.
        These take precedence over random_state if random_state is already present in teacher_kwargs.

    Returns
    -------
    Z : np.ndarray of shape (n_samples, n_components)
        Teacher embedding of X.
    """
    # Strip internal book-keeping keys that aren't valid constructor kwargs
    kw = {k: v for k, v in teacher_kwargs.items()
          if k not in ("save_teacher_model", "save_teacher_path")}
    kw["n_components"] = n_components

    if method == "umap":
        import umap as _umap
        kw.setdefault("random_state", random_state)
        model = _umap.UMAP(**kw)
        Z = model.fit_transform(X)

    elif method == "pca":
        from sklearn.decomposition import PCA
        kw.setdefault("random_state", random_state)
        model = PCA(**kw)
        Z = model.fit_transform(X)

    elif method == "tsne":
        from openTSNE import TSNE
        kw.pop("n_components")          # openTSNE uses n_components differently
        kw.setdefault("random_state", random_state)
        model = TSNE(n_components=n_components, negative_gradient_method="fft", **kw).fit(X)
        Z = model.transform(X)

    elif method == "isomap":
        from sklearn.manifold import Isomap
        # Isomap does not support random_state
        model = Isomap(**kw)
        Z = model.fit_transform(X)

    elif method == "spectral":
        from sklearn.manifold import SpectralEmbedding
        kw.setdefault("random_state", random_state)
        model = SpectralEmbedding(**kw)
        Z = model.fit_transform(X)

    elif method == "phate":
        import phate as _phate
        # phate uses knn instead of n_neighbors; does not use random_state
        # n_components is a constructor argument, not a fit_transform argument
        if "n_neighbors" in kw:
            kw["knn"] = kw.pop("n_neighbors")
        kw.pop("n_components", None)
        model = _phate.PHATE(n_components=n_components, **kw)
        Z = model.fit_transform(X)

    else:
        raise ValueError(f"Unknown teacher method: {method!r}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

    return Z

