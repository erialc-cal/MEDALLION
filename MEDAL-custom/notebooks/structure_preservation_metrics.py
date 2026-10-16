"""
Global and Local Structure Preservation Evaluation Metrics
===========================================================
Implementation based on:
  "Consensus dimension reduction via multi-view learning"
  Bingxue An, Tiffany M. Tang — arXiv:2512.15802

Metrics
-------
Global structure preservation
  1. Random Triplet Accuracy   – proportion of random ordinal triplets preserved
  2. Spearman Correlation      – rank correlation between all pairwise distances

Local structure preservation
  3. LCMC (Local Continuity Meta-Criterion / neighbourhood retention)
     – averaged kNN overlap for small k (default k ∈ {2, 5, 8, 11, 14, 17, 20})
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Return condensed upper-triangular pairwise distance vector."""
    return pdist(X, metric=metric)


def _knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """
    Return an (n, k) array of k-nearest-neighbour indices for each point.
    Self is excluded.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)   # shape (n, k), excludes self by default
    return indices


# ---------------------------------------------------------------------------
# Global Structure Metric 1 — Random Triplet Accuracy
# ---------------------------------------------------------------------------

def random_triplet_accuracy(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_triplets: int = 5_000,
    random_state: int | None = 42,
    metric: str = "euclidean",
) -> float:
    """
    Random Triplet Accuracy (global structure preservation).

    For each randomly sampled anchor–near–far triplet (i, j, k) drawn so that
    d_high(i, j) < d_high(i, k), the metric counts the proportion of triplets
    where the same ordering holds in the low-dimensional embedding:
    d_low(i, j) < d_low(i, k).

    Higher values → better global structure preservation.
    Random baseline ≈ 0.5.

    Parameters
    ----------
    X_high : (n, d_high) array  — original high-dimensional data
    X_low  : (n, d_low)  array  — dimension-reduced embedding
    n_triplets : number of random triplets to sample
    random_state : RNG seed (None for unseeded)
    metric : distance metric passed to scipy.spatial.distance

    Returns
    -------
    accuracy : float in [0, 1]
    """
    rng = np.random.default_rng(random_state)
    n = X_high.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for triplet accuracy.")

    correct = 0
    total = 0
    attempts = 0
    max_attempts = n_triplets * 10  # guard against degenerate datasets

    while total < n_triplets and attempts < max_attempts:
        idx = rng.choice(n, size=3, replace=False)
        i, j, k = idx[0], idx[1], idx[2]

        # distances in high-dimensional space
        d_ij_h = np.linalg.norm(X_high[i] - X_high[j])
        d_ik_h = np.linalg.norm(X_high[i] - X_high[k])

        if d_ij_h == d_ik_h:   # exact tie — skip
            attempts += 1
            continue

        # ensure j is the "near" point, k is the "far" point
        if d_ij_h > d_ik_h:
            j, k = k, j         # swap so d(i,j) < d(i,k) in high-dim

        d_ij_l = np.linalg.norm(X_low[i] - X_low[j])
        d_ik_l = np.linalg.norm(X_low[i] - X_low[k])

        correct += int(d_ij_l < d_ik_l)
        total += 1
        attempts += 1

    if total == 0:
        warnings.warn("No valid triplets sampled; returning NaN.", stacklevel=2)
        return float("nan")

    return correct / total


# ---------------------------------------------------------------------------
# Global Structure Metric 2 — Spearman Correlation of Pairwise Distances
# ---------------------------------------------------------------------------

def spearman_correlation(
    X_high: np.ndarray,
    X_low: np.ndarray,
    metric: str = "euclidean",
    subsample: int | None = None,
    random_state: int | None = 42,
) -> float:
    """
    Spearman rank correlation between pairwise distances in the original
    high-dimensional space and in the embedding (global structure preservation).

    Higher values → better global structure preservation.
    Value of 1 means perfect rank-order preservation of all distances.

    Parameters
    ----------
    X_high : (n, d_high) array
    X_low  : (n, d_low)  array
    metric : distance metric
    subsample : if set, randomly subsample this many points before computing
                (useful for large n where O(n²) pairs are expensive)
    random_state : RNG seed

    Returns
    -------
    rho : float in [-1, 1]
    """
    n = X_high.shape[0]

    if subsample is not None and subsample < n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=subsample, replace=False)
        X_high = X_high[idx]
        X_low = X_low[idx]

    d_high = _pairwise_distances(X_high, metric=metric)
    d_low = _pairwise_distances(X_low, metric=metric)

    rho, _ = spearmanr(d_high, d_low)
    return float(rho)


# ---------------------------------------------------------------------------
# Local Structure Metric 3 — LCMC (neighbourhood retention)
# ---------------------------------------------------------------------------

def lcmc(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k_values: Sequence[int] = range(2, 21, 3),
) -> dict[int, float]:
    """
    Local Continuity Meta-Criterion (LCMC) — local structure preservation.

    For each k, measures the average proportion of k-nearest neighbours that
    are shared between the high- and low-dimensional spaces.  Sometimes called
    "neighbourhood retention."

    LCMC(k) = (1/n) * Σᵢ  |kNN_high(i) ∩ kNN_low(i)| / k

    Higher values → better local structure preservation.  Maximum value is 1.

    Parameters
    ----------
    X_high   : (n, d_high) array
    X_low    : (n, d_low)  array
    k_values : iterable of neighbourhood sizes to evaluate
               (paper uses k ∈ {2, 5, 8, 11, 14, 17, 20})

    Returns
    -------
    scores : dict mapping each k to its LCMC score
    """
    k_values = sorted(set(k_values))
    k_max = max(k_values)

    n = X_high.shape[0]
    if k_max >= n:
        raise ValueError(f"k_max ({k_max}) must be < n ({n}).")

    knn_high = _knn_indices(X_high, k_max)   # (n, k_max)
    knn_low = _knn_indices(X_low, k_max)     # (n, k_max)

    # convert rows to sets for intersection
    high_sets = [set(knn_high[i]) for i in range(n)]
    low_sets = [set(knn_low[i]) for i in range(n)]

    scores: dict[int, float] = {}
    for k in k_values:
        overlap = 0.0
        for i in range(n):
            # only consider the first k neighbours (sorted by distance)
            h_k = set(knn_high[i, :k])
            l_k = set(knn_low[i, :k])
            overlap += len(h_k & l_k)
        scores[k] = overlap / (n * k)

    return scores


def lcmc_mean(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k_values: Sequence[int] = range(2, 21, 3),
) -> float:
    """
    Mean LCMC score averaged over all k in k_values.
    This is the summary scalar reported in the paper's Figure 6.
    """
    scores = lcmc(X_high, X_low, k_values=k_values)
    return float(np.mean(list(scores.values())))


# ---------------------------------------------------------------------------
# Convenience: evaluate all metrics at once
# ---------------------------------------------------------------------------

def evaluate_embedding(
    X_high: np.ndarray,
    X_low: np.ndarray,
    *,
    n_triplets: int = 5_000,
    k_values: Sequence[int] = range(2, 21, 3),
    spearman_subsample: int | None = 2_000,
    random_state: int | None = 42,
    verbose: bool = True,
) -> dict:
    """
    Compute all global and local structure preservation metrics.

    Parameters
    ----------
    X_high            : (n, d_high) original high-dimensional data
    X_low             : (n, d_low)  dimension-reduced embedding to evaluate
    n_triplets        : number of random triplets for triplet accuracy
    k_values          : neighbourhood sizes for LCMC
    spearman_subsample: subsample size for Spearman (None = use all n points)
    random_state      : RNG seed
    verbose           : print a summary table

    Returns
    -------
    results : dict with keys
        'triplet_accuracy'   – float
        'spearman_rho'       – float
        'lcmc'               – dict {k: score}
        'lcmc_mean'          – float
    """
    rta = random_triplet_accuracy(
        X_high, X_low, n_triplets=n_triplets, random_state=random_state
    )
    rho = spearman_correlation(
        X_high, X_low, subsample=spearman_subsample, random_state=random_state
    )
    lcmc_scores = lcmc(X_high, X_low, k_values=k_values)
    lcmc_avg = float(np.mean(list(lcmc_scores.values())))

    results = {
        "triplet_accuracy": rta,
        "spearman_rho": rho,
        "lcmc": lcmc_scores,
        "lcmc_mean": lcmc_avg,
    }

    if verbose:
        print("=" * 52)
        print("  Structure Preservation Evaluation  (arXiv:2512.15802)")
        print("=" * 52)
        print(f"\n── Global Structure ──────────────────────────────────")
        print(f"  Random Triplet Accuracy : {rta:.4f}  (baseline ≈ 0.50)")
        print(f"  Spearman ρ              : {rho:.4f}  (range −1 to 1)")
        print(f"\n── Local Structure ───────────────────────────────────")
        print(f"  LCMC mean (k∈{list(k_values)}) : {lcmc_avg:.4f}")
        for k, s in lcmc_scores.items():
            print(f"    k={k:2d}  LCMC = {s:.4f}")
        print("=" * 52)

    return results


# ---------------------------------------------------------------------------
# Demo / usage example
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     import sklearn.datasets as ds
#     from sklearn.decomposition import PCA
#     from sklearn.manifold import TSNE

#     print("\n>>> Loading Swiss Roll dataset (n=1000, d=3) ...")
#     X, _ = ds.make_swiss_roll(n_samples=1_000, noise=0.1, random_state=0)

#     print(">>> Embedding with PCA (good global structure preservation) ...")
#     X_pca = PCA(n_components=2, random_state=0).fit_transform(X)

#     print(">>> Embedding with t-SNE (good local structure preservation) ...")
#     X_tsne = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(X)

#     print("\n\n========  PCA  ========")
#     evaluate_embedding(X, X_pca, verbose=True)

#     print("\n\n========  t-SNE  ========")
#     evaluate_embedding(X, X_tsne, verbose=True)
