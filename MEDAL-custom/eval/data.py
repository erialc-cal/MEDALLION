"""
Dataset loading for MEDAL paper experiments.

All datasets are stored under PATH_PREFIX (defined in configs.py).
Each loader returns (X, labels) where X is float32, zero-indexed,
and labels may be None if not available or not requested.

Usage
-----
    from eval.data import load_dataset
    X, labels = load_dataset("mnist")
    X, _      = load_dataset("macaque")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional

from eval.configs import PATH_PREFIX

# Datasets that require StandardScaler normalisation
_NEEDS_SCALING = {"wine", "gene_cancer", "astro"}


def load_dataset(
    dataset_name: str,
    labels: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a full dataset as a float32 array.

    Parameters
    ----------
    dataset_name : str
        One of: "mnist", "wine", "gene_cancer", "darmanis", "hydra",
        "astro", "tasic", "macaque".
    labels : bool
        If True, also return class/cluster labels.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features), float32
    labs : np.ndarray or None
        Class labels if ``labels=True``, else None.
    """
    X, labs = _load_raw(dataset_name)

    if dataset_name in _NEEDS_SCALING:
        X = StandardScaler().fit_transform(X)

    X = X.astype(np.float32)

    return (X, labs) if labels else (X, None)


def load_and_split(
    dataset_name: str,
    test_size: float = 0.2,
    seed: int = 0,
    labels: bool = False,
) -> tuple:
    """
    Load a dataset and return a reproducible train/test split.

    Parameters
    ----------
    dataset_name : str
    test_size : float
        Fraction held out as test set.
    seed : int
        Random seed for the split.
    labels : bool
        If True, returns (X_train, X_test, labs_train, labs_test).
        If False, returns (X_train, X_test).

    Returns
    -------
    X_train, X_test : np.ndarray
    labs_train, labs_test : np.ndarray  (only when labels=True)
    """
    X, labs = load_dataset(dataset_name, labels=labels)

    if labels and labs is not None:
        X_train, X_test, labs_train, labs_test = train_test_split(
            X, labs, test_size=test_size, random_state=seed
        )
        return X_train, X_test, labs_train, labs_test

    X_train, X_test = train_test_split(X, test_size=test_size, random_state=seed)
    return X_train, X_test


# ---------------------------------------------------------------------------
# Internal per-dataset loaders
# ---------------------------------------------------------------------------

def _load_raw(dataset_name: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    p = Path(PATH_PREFIX)

    if dataset_name == "mnist":
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X    = mnist.data[:10000].astype(np.float32) / 255.0
        labs = mnist.target[:10000]
        return X, labs

    if dataset_name == "wine":
        from sklearn.datasets import load_wine
        data = load_wine()
        return data.data.astype(np.float32), data.target

    if dataset_name == "gene_cancer":
        df = pd.read_csv(p / "PANCAN-801x20531" / "data.csv", index_col=0)
        labs = pd.read_csv(p / "PANCAN-801x20531" / "labels.csv", index_col=0).values.flatten()
        return df.values.astype(np.float32), labs

    if dataset_name == "darmanis":
        df   = pd.read_csv(p / "GBM_HVG500_with_metadata.csv", index_col=0)
        X    = df.iloc[:, 29:].to_numpy(dtype=np.float32)
        labs = df["Location"].values
        return X, labs

    if dataset_name == "hydra":
        df   = pd.read_csv(p / "Hydra500_official.csv")
        labs = pd.read_csv(p / "Hydra_labels.csv")["cluster.manuscript"].values
        X    = df.drop("labels", axis=1).to_numpy(dtype=np.float32)
        return X, labs

    if dataset_name == "astro":
        X    = pd.read_csv(p / "data_mean_imputed_with_ids_all.csv", index_col=0).to_numpy(dtype=np.float32)
        labs = pd.read_csv(p / "cluster_labels_final.csv", index_col=0).to_numpy().flatten()
        return X, labs

    if dataset_name == "tasic":
        X    = np.load(p / "preprocessed-data.npy").astype(np.float32)
        labs = np.load(p / "tasic_cluster_labels.npy", allow_pickle=True)
        return X, labs

    if dataset_name == "macaque":
        df   = pd.read_csv(p / "macaque1_pc100.csv")
        labs = df["labels"].values
        X    = df.drop("labels", axis=1).to_numpy(dtype=np.float32)
        return X, labs
    
    if dataset_name == "macaque2":
        df   = pd.read_csv(p / "macaque2_pc100.csv")
        labs = df["labels"].values
        X    = df.drop("labels", axis=1).to_numpy(dtype=np.float32)
        return X, labs
    
    if dataset_name == "macaque3":
        df   = pd.read_csv(p / "macaque3_pc100.csv")
        labs = df["labels"].values
        X    = df.drop("labels", axis=1).to_numpy(dtype=np.float32)
        return X, labs

    raise ValueError(
        f"Unknown dataset {dataset_name!r}. "
        f"Known: mnist, wine, gene_cancer, darmanis, hydra, astro, tasic, macaque."
    )