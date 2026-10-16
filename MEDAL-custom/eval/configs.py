"""
Experiment configurations used in the MEDAL paper.

All dataset-specific hyperparameters (architecture, training, teacher sweeps)
are defined here as the single source of truth for reproducing paper results.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

PATH_PREFIX = "/share/ctn/users/bnc2119/drd_data"

# ---------------------------------------------------------------------------
# Distillation convergence bands
# Each entry is a list of (lower, upper) loss bands used for early stopping.
# ---------------------------------------------------------------------------

DISTILL_BANDS_DICT = {
    "gene_cancer": [(1e-12, 9e-6)],
    "mnist":       [(1e-12, 9e-6)],
    "darmanis":    [(1e-12, 9e-6)],
    "wine":        [(1e-12, 9e-8)],
    "hydra":       [(1e-12, 9e-6)],
    "astro":       [(1e-12, 9e-6)],
    "tasic":       [(1e-12, 9e-6)],
    "macaque":     [(1e-12, 9e-6)],
}

# ---------------------------------------------------------------------------
# Architecture + training config per dataset  (arch_config for run_teacher_sweep)
# ---------------------------------------------------------------------------

INIT_CONFIG = {
    "gene_cancer": {
        "lr":                   1e-3,
        "lambda_d":             10000,
        "eta_min":              1e-11,
        "hidden_dims":          [1000, 1000, 1000, 1000],
        "activation":           "SELU",
        "bottleneck_activation": None,
        "max_epochs":           7000,
        "warmup":               0,
        "batch_size":           100,
        "t_factor":             0.95,
        "t_patience":           20,
        "use_batchnorm":        True,
        "test_size":            0.2,
    },
    "wine": {
        "lr":                   0.02,
        "lambda_d":             20000,
        "eta_min":              1e-8,
        "hidden_dims":          [258, 258, 258, 258],
        "activation":           "SELU",
        "bottleneck_activation": None,
        "max_epochs":           130000,
        "warmup":               0,
        "batch_size":           100,
    },
    "mnist": {
        "lr":                   1e-3,
        "lambda_d":             3000,
        "eta_min":              1e-7,
        "hidden_dims":          [512, 512, 512, 512],
        "activation":           "SELU",
        "bottleneck_activation": None,
        "max_epochs":           5000,
        "batch_size":           256,
        "warmup":               0,
        "t_patience":           20,
        "use_batchnorm":        False,
        "test_size":            0.2,
    },
    "darmanis": {
        "lr":                   0.001,
        "lambda_d":             50000,
        "eta_min":              1e-7,
        "hidden_dims":          [1000, 1000, 1000, 1000],
        "activation":           "SELU",
        "bottleneck_activation": None,
        "max_epochs":           20000,
        "warmup":               0,
        "batch_size":           10000,
        "t_factor":             0.95,
        "t_patience":           20,
        "use_batchnorm":        False,
        "test_size":            0.2,
    },
    "hydra": {
        "lr":                   0.005,
        "lambda_d":             30000,
        "eta_min":              1e-7,
        "hidden_dims":          [256, 1024, 1024, 1024],
        "activation":           "SELU",
        "bottleneck_activation": None,
        "max_epochs":           20000,
        "warmup":               1500,
        "batch_size":           51200,
        "t_factor":             0.95,
        "t_patience":           20,
        "use_batchnorm":        True,
        "test_size":            0.2,
    },
    "astro": {
        "lr":                   0.0005,
        "lambda_d":             10000,
        "eta_min":              1e-7,
        "hidden_dims":          [512, 512, 512, 512],
        "activation":           "SELU",
        "bottleneck_activation": None,
        "max_epochs":           60000,
        "warmup":               0,
        "batch_size":           25600,
        "t_factor":             0.9,
        "t_patience":           70,
        "use_batchnorm":        False,
        "test_size":            0.2,
    },
    "tasic": {
        "lr":                   0.00268681,
        "lambda_d":             10000,
        "eta_min":              1e-7,
        "hidden_dims":          [256, 1024, 1024, 1024],
        "activation":           "SELU",
        "bottleneck_activation": None,
        "max_epochs":           6000,
        "warmup":               100,
        "batch_size":           50000,
        "t_factor":             0.95,
        "t_patience":           20,
        "use_batchnorm":        True,
        "test_size":            0.2,
    },
    "macaque": {
        "lr":                   0.0005,
        "lambda_d":             10000,
        "eta_min":              1e-7,
        "hidden_dims":          [512, 512, 512, 512],
        "activation":           "SELU",
        "bottleneck_activation": None,
        "max_epochs":           25000,
        "warmup":               0,
        "batch_size":           1024,
        "t_factor":             0.95,
        "t_patience":           70,
        "use_batchnorm":        True,
        "test_size":            0.2,
    },
    
}

INIT_CONFIG["macaque2"] = INIT_CONFIG["macaque"]

# ---------------------------------------------------------------------------
# Teacher hyperparameter sweep grids
# Keys use t_n_neighbors (translated to n_neighbors at runtime in run_sweep.py).
# ---------------------------------------------------------------------------

TEACHER_SWEEP_SPECS = {
    "mnist": {
        "umap": {
            "t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int)).tolist(),
            "min_dist":      [0.1],
            "n_components":  [2],
        },
        "tsne": {
            "perplexity":   [5, 11, 27, 62, 146, 341, 793, 1846],
            "n_components": [2],
        },
        "spectral": {
            "t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int)).tolist(),
            "n_components":  [2],
        },
        "phate": {
            "t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int)).tolist(),
            "n_components":  [2],
        },
        "pca": {"n_components": [2]},
    },
    "hydra": {
        "umap": {
            "t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(2000), 10).astype(int)).tolist(),
            "min_dist":      [0.1],
            "n_components":  [2],
        },
        "tsne": {
            "perplexity":   np.unique(np.logspace(np.log10(5), np.log10(5000), 10).astype(int)).tolist(),
            "n_components": [2],
        },
        "spectral": {
            "t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(200), 15).astype(int)).tolist(),
            "n_components":  [2],
        },
        "pca": {"n_components": [2]},
    },
    "tasic": {
        "umap": {
            "t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(2000), 10).astype(int)).tolist(),
            "min_dist":      [0.1],
            "n_components":  [2],
        },
        "tsne": {
            "perplexity":   np.unique(np.logspace(np.log10(5), np.log10(6000), 10).astype(int)).tolist(),
            "n_components": [2],
        },
        "spectral": {
            "t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(200), 15).astype(int)).tolist(),
            "n_components":  [2],
        },
        "pca": {"n_components": [2]},
    },
    "astro": {
        "umap": {
            "t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int)).tolist(),
            "min_dist":      [0.1],
            "n_components":  [2],
        },
        "tsne": {
            "perplexity":   np.unique(np.logspace(np.log10(3), np.log10(500), 15).astype(int)).tolist(),
            "n_components": [2],
        },
        "spectral": {
            "t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int)).tolist(),
            "n_components":  [2],
        },
        "pca": {"n_components": [2]},
    },
    "macaque": {
        "umap": {
            "t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int)).tolist(),
            "min_dist":      [0.1],
            "n_components":  [2],
        },
        "tsne": {
            "perplexity":   np.unique(np.logspace(np.log10(3), np.log10(500), 15).astype(int)).tolist(),
            "n_components": [2],
        },
        "spectral": {
            "t_n_neighbors": np.unique(np.logspace(np.log10(5), np.log10(500), 15).astype(int)).tolist(),
            "n_components":  [2],
        },
        "pca": {"n_components": [2]},
    },
}

TEACHER_SWEEP_SPECS["macaque2"] = TEACHER_SWEEP_SPECS["macaque"]

# ---------------------------------------------------------------------------
# Rank sweep: fix teacher hyperparameter, sweep latent dimensionality
# ---------------------------------------------------------------------------

RANK_SWEEP_SPECS = {
    "mnist": {
        "umap": {
            "t_n_neighbors": [18],
            "min_dist":      [0.1],
            "n_components":  np.unique(np.logspace(np.log10(2), np.log10(100), 10).astype(int)).tolist(),
        },
        "spectral": {
            "t_n_neighbors": [18],
            "n_components":  np.unique(np.logspace(np.log10(2), np.log10(100), 10).astype(int)).tolist(),
        },
        "pca": {
            "n_components":  np.unique(np.logspace(np.log10(2), np.log10(500), 15).astype(int)).tolist(),
        },
    },
    "gene_cancer": {
        "umap": {
            "t_n_neighbors": [5],
            "min_dist":      [0.1],
            "n_components":  np.unique(np.logspace(np.log10(2), np.log10(100), 10).astype(int)).tolist(),
        },
        "spectral": {
            "t_n_neighbors": [5],
            "n_components":  np.unique(np.logspace(np.log10(2), np.log10(100), 10).astype(int)).tolist(),
        },
        "pca": {
            "n_components":  np.unique(np.logspace(np.log10(2), np.log10(100), 10).astype(int)).tolist(),
        },
    },
}
