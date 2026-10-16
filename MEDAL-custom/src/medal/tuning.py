"""
AE architecture search via Ray Tune.

Primary API::

    from medal.tuning import tune_medal_architecture

    result = tune_medal_architecture(
        X_train,
        teacher="umap",
        output_dir="experiments/arch_search",
    )
    # result.best_config is ready to pass to run_teacher_sweep
    print(result)

Lower-level API (for custom search spaces)::

    from medal.tuning import tune_architecture, get_best_config

    analysis = tune_architecture(
        X_train,
        output_dir="experiments/arch_search",
        teacher="tsne",
        teacher_params={"perplexity": 30},
    )
    arch_config = get_best_config(analysis)
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

from medal.teacher import get_teacher_embeddings
from medal.normalizer import GlobalEmbeddingNormalizer
# Modified: added for custom teacher support + parameter sweep
from medal.custom_teacher import (TeacherSpec, compute_embedding, is_custom_key)
from medal._provenance import atomic_json, guard_custom_output

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Keys injected by the tuning functions that are NOT architecture params
_INTERNAL_KEYS = frozenset({
    "_arch_search", "_emb_path", "_norm_path",
    "output_dir", "dataset_name", "teacher", "teacher_config",
    "seed", "normalize_teacher", "latent_dim",
    "verbose", "distill_bands",
})

# Sensible teacher defaults when teacher_params is not provided
_DEFAULT_TEACHER_PARAMS: Dict[str, dict] = {
    "umap":     {"n_neighbors": 15, "min_dist": 0.1},
    "tsne":     {"perplexity": 30},
    "pca":      {},
    "spectral": {"n_neighbors": 15},
    "phate":    {"n_neighbors": 15},
    "isomap":   {"n_neighbors": 15},
}


# ------------------------------------------------------------------
# ArchSearchResults
# ------------------------------------------------------------------

@dataclass
class ArchSearchResults:
    """
    Structured result returned by :func:`tune_medal_architecture`.

    Attributes
    ----------
    best_config : dict
        Winning architecture config, cleaned of all Ray / internal keys.
        Ready to pass directly to :func:`~medal.sweep.run_teacher_sweep`
        as arch_config.
    best_metrics : dict
        Last reported metrics from the best trial
        (e.g. distill_loss, recon_loss, lr).
    results_df : pd.DataFrame
        One row per trial with config columns and final metrics.
    teacher_emb_path : Path
        Path to the pre-computed teacher embedding (.npy).
    output_dir : Path
        Root directory where all outputs were written.
    metric : str
        Metric that was optimised (default "distill_loss").
    mode : str
        "min" or "max".
    """
    best_config: Dict[str, Any]
    best_metrics: Dict[str, Any]
    results_df: pd.DataFrame
    teacher_emb_path: Path
    output_dir: Path
    metric: str = "distill_loss"
    mode: str = "min"

    def __post_init__(self):
        self.teacher_emb_path = Path(self.teacher_emb_path)
        self.output_dir = Path(self.output_dir)

    # ------------------------------------------------------------------

    def to_arch_config(self) -> dict:
        """
        Return the architecture config dict for passing to
        :func:`~medal.sweep.run_teacher_sweep`.

        This is an alias for self.best_config; provided for
        discoverability when chaining calls::

            results = tune_medal_architecture(X_train, ...)
            sweep   = run_teacher_sweep(X_train, ..., arch_config=results.to_arch_config())
        """
        return dict(self.best_config)

    def save(self, path: Optional[str | Path] = None) -> Path:
        """
        Write results to disk.

        Saves two files:

        * arch_search_results.csv — full per-trial results table.
        * best_config.json — best architecture + optimisation config,
          ready to pass as arch_config to :func:`~medal.sweep.run_teacher_sweep`.

        Parameters
        ----------
        path : str or Path, optional
            Destination for the CSV.  Defaults to
            output_dir/arch_search_results.csv.  The JSON is always
            written to output_dir/best_config.json.

        Returns
        -------
        Path
            Path to the CSV file.
        """
        dest = Path(path) if path else self.output_dir / "arch_search_results.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.results_df.to_csv(dest, index=False)

        meta = {
            "best_config":      _serialisable(self.best_config),
            "best_metrics":     _serialisable(self.best_metrics),
            "teacher_emb_path": str(self.teacher_emb_path),
            "metric":           self.metric,
            "mode":             self.mode,
        }
        json_path = self.output_dir / "best_config.json"
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        return dest

    @classmethod
    def load(cls, output_dir: str | Path) -> "ArchSearchResults":
        """
        Reload a previously saved :class:`ArchSearchResults` from disk.

        Reads best_config.json and arch_search_results.csv from
        *output_dir*.

        Parameters
        ----------
        output_dir : str or Path
            The directory passed as output_dir to
            :func:`tune_medal_architecture`.

        Returns
        -------
        ArchSearchResults

        Example
        -------
        ::
            result = medal.ArchSearchResults.load("output/arch_search")
            sweep  = medal.run_teacher_sweep(
                X_train,
                output_dir="output/teacher_sweep",
                teacher="umap",
                arch_config=result.best_config,
            )
        """
        output_dir = Path(output_dir)
        json_path  = output_dir / "best_config.json"
        csv_path   = output_dir / "arch_search_results.csv"

        if not json_path.exists():
            raise FileNotFoundError(
                f"best_config.json not found in {output_dir}. "
                "Re-run tune_medal_architecture() with save_results=True."
            )

        with open(json_path) as f:
            meta = json.load(f)

        results_df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

        return cls(
            best_config=meta["best_config"],
            best_metrics=meta["best_metrics"],
            results_df=results_df,
            teacher_emb_path=Path(meta["teacher_emb_path"]),
            output_dir=output_dir,
            metric=meta.get("metric", "distill_loss"),
            mode=meta.get("mode", "min"),
        )

    def __repr__(self) -> str:
        m = self.best_metrics
        loss_str = f"{m.get(self.metric, '?'):.4e}" if isinstance(m.get(self.metric), float) else "?"
        hd = self.best_config.get("hidden_dims", "?")
        lr = self.best_config.get("lr", "?")
        ld = self.best_config.get("lambda_d", "?")
        return (
            f"ArchSearchResults(\n"
            f"  {self.metric}={loss_str}\n"
            f"  hidden_dims={hd},  lr={lr},  lambda_d={ld}\n"
            f"  output_dir={self.output_dir}\n"
            f")"
        )


# ------------------------------------------------------------------
# Primary user-facing API
# ------------------------------------------------------------------

def tune_medal_architecture(
    X: np.ndarray,
    teacher: str = "umap",
    teacher_params: Optional[dict] = None,
    output_dir: Optional[str | Path] = None,
    latent_dim: int = 2,
    search_space: Optional[dict] = None,
    search_mode: str = "grid",
    num_samples: int = 1,
    resources_per_trial: Optional[dict] = None,
    metric: str = "distill_loss",
    mode: str = "min",
    scheduler=None,
    ray_storage_path: Optional[str] = None,
    seed: int = 0,
    test_size: float = 0.2,
    max_epochs: int = 3000,
    save_results: bool = True,
    verbose: bool = True,
) -> ArchSearchResults:
    """
    Search for the best MEDAL autoencoder architecture using Ray Tune.

    A teacher embedding is computed once from a fixed set of teacher
    hyperparameters, then every candidate architecture is trained to
    distil it.  The winner is returned in an ArchSearchResults
    object whose best_config can be passed directly to
    medal.sweep.run_teacher_sweep.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training data (will be split internally into train / val).
    teacher : str
        Teacher algorithm — "umap", "tsne", "pca",
        "spectral", "phate", or "isomap".
    teacher_params : dict, optional
        Teacher hyperparameters (e.g. {"perplexity": 30}).
        Sensible defaults are used when None.
    output_dir : str or Path, optional
        Directory for embeddings, checkpoints, and Ray results.
        Defaults to ./medal_arch_search.
    latent_dim : int
        Target embedding dimensiona (passed to the teacher and
        used as the AE bottleneck size).
    search_space : dict, optional
        Ray Tune parameter specs to *override* individual keys in the
        default search space.  Use tune.grid_search /
        tune.choice / tune.loguniform etc. directly::

            from ray import tune
            search_space = {"lambda_d": tune.grid_search([10, 100, 1000])}

    search_mode : {"grid", "random"}
        "grid"   — exhaustive grid over the default space
                       (use num_samples=1).
        "random" — random sampling; set num_samples to the
                       desired number of trials (e.g. 20–50).
    num_samples : int
        Number of Ray Tune samples.  For "grid" mode, leave at 1
        (the grid is the source of trials).  For "random" mode,
        this controls how many configs to sample.
    resources_per_trial : dict, optional
        E.g. {"cpu": 4, "gpu": 1}.  Defaults to one GPU per trial.
    metric : str
        Metric to optimise — "distill_loss" (default) or
        "recon_loss".
    mode : str
        "min" (default) or "max".
    scheduler : ray.tune scheduler, optional
        Pass a pre-constructed Ray Tune scheduler such as
        AsyncHyperBandScheduler(...) to enable early stopping.
        None (default) runs every trial to completion.
    ray_storage_path : str, optional
        Override Ray Tune's storage path for trial checkpoints.
        Defaults to output_dir/ray_results.
    seed : int
        Random seed used for the train/val split.
    test_size : float
        Fraction of *X* held out as a validation set.
    max_epochs : int
        Maximum training epochs per trial.
    save_results : bool
        If True, write arch_search_results.csv to output_dir.
    verbose : bool
        Print progress and leaderboard.

    Returns
    -------
    ArchSearchResults
        Structured result object.  result.best_config is ready for
        ~medal.sweep.run_teacher_sweep.

    Examples
    --------
    >>> result = tune_medal_architecture(X_train, teacher="tsne",
    ...     teacher_params={"perplexity": 30}, output_dir="exp/arch")
    >>> result.best_config
    {'hidden_dims': [512, 512, 512], 'lr': 0.001, 'lambda_d': 1000, ...}
    """
    from sklearn.model_selection import train_test_split

    # ── output directory ────────────────────────────────────────────
    output_dir = Path(output_dir).resolve() if output_dir else Path("medal_arch_search").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- custom teacher's specifications and parameters --------------
    teacher_spec = teacher if isinstance(teacher, TeacherSpec) else None

    if teacher_spec is None and is_custom_key(teacher):
        raise ValueError("Pass the TeacherSpec, not its internal key.")

    if teacher_spec is not None:
        resolved = teacher_spec.resolve_params(
            teacher_params,
            n_components=latent_dim,
        )
        teacher_params = resolved
        teacher = teacher_spec.key

        guard_custom_output(
            output_dir,
            X,
            {
                "kind": "architecture_search",
                "teacher": teacher_spec.metadata(),
                "teacher_params": resolved,
                "latent_dim": latent_dim,
                "seed": seed,
                "test_size": test_size,
                "max_epochs": max_epochs,
                "search_mode": search_mode,
                "num_samples": num_samples,
            },
        )
        atomic_json(
            Path(output_dir) / "teacher_metadata.json",
            teacher_spec.metadata(),
        )

    # ── teacher params ──────────────────────────────────────────────
    if teacher_params is None:
        teacher_params = dict(_DEFAULT_TEACHER_PARAMS.get(teacher, {}))
        if verbose:
            print(f"[tune_medal_architecture] Using default teacher params for "
                  f"{teacher!r}: {teacher_params}")

    n_components = teacher_params.get("n_components", latent_dim)
    teacher_kw   = {k: v for k, v in teacher_params.items() if k != "n_components"}

    # ── train / val split ───────────────────────────────────────────
    if teacher_spec is not None:
        train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=test_size, random_state=seed)
        X_train = X[train_idx]
        X_val = X[val_idx]
        np.save(output_dir / "idx_train.npy", train_idx)
        np.save(output_dir / "idx_val.npy", val_idx)
    else:
        X_train, X_val = train_test_split(X, test_size=test_size, random_state=seed)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy", X_val)
    if verbose:
         print(f"[tune_medal_architecture] Train: {X_train.shape}, "
               f"Val: {X_val.shape}")
        
    # ── teacher embedding (computed once) ───────────────────────────
    if verbose:
        print(f"[tune_medal_architecture] Computing {teacher} embedding …")
    # Z_raw      = get_teacher_embeddings(teacher, X_train, n_components=n_components, **teacher_kw)
    Z_raw = compute_embedding(
        teacher_spec if teacher_spec is not None else teacher,
        X_train,
        n_components=n_components,
        **teacher_kw,
    )
    normalizer = GlobalEmbeddingNormalizer().fit(Z_raw)

    emb_dir   = output_dir / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    emb_path  = emb_dir / "arch_search_teacher.npy"
    norm_path = emb_dir / "arch_search_teacher.norm.pkl"
    np.save(emb_path, Z_raw)
    normalizer.save(norm_path)

    if verbose:
        print(f"[tune_medal_architecture] Embedding saved to {emb_path}")

    # ── search space ────────────────────────────────────────────────
    space = _build_arch_search_space(search_mode)
    if search_space:
        space.update(search_space)

    # ── fixed training hyperparameters ──────────────────────────────
    fixed = {
        "max_epochs":         max_epochs,
        "batch_size":         256,
        "warmup":             0,
        "eta_min":            1e-7,
        "adamw_weight_decay": 1e-5,
        "t_factor":           0.9,
        "t_patience":         20,
        "distill_bands":      None,
        "bottleneck_activation": None,
        "final_activation":   None,
        "dropout_rate":       0.0,
        "verbose":            False,
    }

    # ── full Ray Tune config ─────────────────────────────────────────
    config = {
        **fixed,
        **space,
        # Internal keys consumed by _arch_search_trainable
        "_arch_search":      True,
        "output_dir":        str(output_dir),
        "dataset_name":      "data",
        "teacher":           teacher,
        "latent_dim":        latent_dim,
        "teacher_config":    {**teacher_params, "n_components": n_components},
        "seed":              seed,
        "normalize_teacher": True,
        "_emb_path":         str(emb_path),
        "_norm_path":        str(norm_path),
    }

    # ── run ─────────────────────────────────────────────────────────
    storage = ray_storage_path or str(output_dir / "ray_results")

    analysis = tune.run(
        _arch_search_trainable,
        name="medal_arch_search",
        num_samples=num_samples,
        resources_per_trial=resources_per_trial or {"cpu": 4, "gpu": 1},
        config=config,
        metric=metric,
        mode=mode,
        verbose=1 if verbose else 0,
        max_failures=3,
        scheduler=scheduler,
        storage_path=storage,
    )

    # ── extract results ──────────────────────────────────────────────
    best_config  = _clean_config(analysis.get_best_config(metric, mode))
    best_trial   = analysis.get_best_trial(metric, mode)
    best_metrics = dict(best_trial.last_result) if best_trial else {}

    results_df = analysis.results_df.copy()

    result = ArchSearchResults(
        best_config=best_config,
        best_metrics=best_metrics,
        results_df=results_df,
        teacher_emb_path=emb_path,
        output_dir=output_dir,
        metric=metric,
        mode=mode,
    )

    if save_results:
        csv_path = result.save()
        if verbose:
            print(f"[tune_medal_architecture] Results saved to {csv_path}")

    if verbose:
        _print_leaderboard(analysis, metric=metric, top_k=5)
        print(f"\n{result}")

    return result


# ------------------------------------------------------------------
# Lower-level API (kept for backward compatibility)
# ------------------------------------------------------------------

def tune_architecture(
    X: np.ndarray,
    output_dir: str | Path,
    teacher: str,
    teacher_params: dict,
    latent_dim: int = 2,
    search_space: Optional[dict] = None,
    val_size: float = 0.2,
    num_samples: int = 1,
    resources_per_trial: Optional[dict] = None,
    scheduler: str = "ahb",
    base_config: Optional[dict] = None,
    verbose: bool = True,
):
    """
    Search for the best autoencoder architecture using Ray Tune.

    .. deprecated::
        Prefer :func:`tune_medal_architecture` for new code.

    Parameters
    ----------
    X : np.ndarray
    output_dir : str or Path
    teacher : str
    teacher_params : dict
        A single set of teacher hyperparameters,
        e.g. {"perplexity": 30, "n_components": 2}.
    latent_dim : int
    search_space : dict, optional
        Merged on top of (and overrides) the default search space.
    val_size : float
    num_samples : int
    resources_per_trial : dict, optional
    scheduler : str
        "ahb" (AsyncHyperBand) or None.
    base_config : dict, optional
        Base MEDAL training config overrides.
    verbose : bool

    Returns
    -------
    analysis : ray.tune.ExperimentAnalysis
        Pass to :func:`get_best_config` to extract the winning architecture.
    """
    from sklearn.model_selection import train_test_split

    if isinstance(teacher, TeacherSpec):
        raise TypeError(
            "Custom TeacherSpec is supported by "
            "tune_medal_architecture(); use that API."
        )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val = train_test_split(X, test_size=val_size, random_state=0)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy",   X_val)

    n_components = teacher_params.get("n_components", latent_dim)
    kw = {k: v for k, v in teacher_params.items() if k != "n_components"}
    Z_raw      = get_teacher_embeddings(teacher, X_train,
                                        n_components=n_components, **kw)
    normalizer = GlobalEmbeddingNormalizer().fit(Z_raw)
    emb_dir    = output_dir / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    emb_path   = emb_dir / "arch_search_teacher.npy"
    norm_path  = emb_dir / "arch_search_teacher.norm.pkl"
    np.save(emb_path, Z_raw)
    normalizer.save(norm_path)

    defaults = {
        "lambda_d":          10,
        "lr":                1e-3,
        "max_epochs":        3000,
        "batch_size":        256,
        "warmup":            0,
        "eta_min":           1e-7,
        "adamw_weight_decay":1e-5,
        "t_factor":          0.9,
        "t_patience":        20,
        "distill_bands":     None,
        "verbose":           False,
    }
    if base_config:
        defaults.update(base_config)

    space = _legacy_default_search_space(X_train.shape[1])
    if search_space:
        space.update(search_space)

    config = {
        **defaults,
        **space,
        "_arch_search":      True,
        "output_dir":        str(output_dir),
        "dataset_name":      "data",
        "teacher":           teacher,
        "latent_dim":        latent_dim,
        "teacher_config":    {**teacher_params, "n_components": n_components},
        "seed":              0,
        "normalize_teacher": True,
        "_emb_path":         str(emb_path),
        "_norm_path":        str(norm_path),
    }

    sched = None
    if scheduler == "ahb":
        sched = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="distill_loss",
            mode="min",
            grace_period=500,
            max_t=3000,
        )

    analysis = tune.run(
        _arch_search_trainable,
        name="medal_arch_search",
        num_samples=num_samples,
        resources_per_trial=resources_per_trial or {"cpu": 4, "gpu": 1},
        config=config,
        verbose=1 if verbose else 0,
        max_failures=3,
        scheduler=sched,
        storage_path=str(output_dir / "ray_results"),
    )

    if verbose:
        _print_leaderboard(analysis, top_k=5)

    return analysis


def get_best_config(
    analysis,
    metric: str = "distill_loss",
    mode: str = "min",
    top_k: int = 5,
) -> dict:
    """
    Extract the best architecture config from a tune_architecture result.

    Parameters
    ----------
    analysis : ray.tune.ExperimentAnalysis
    metric : str
    mode : str
    top_k : int

    Returns
    -------
    dict
        Architecture config ready for medal.sweep.run_teacher_sweep.
    """
    _print_leaderboard(analysis, metric=metric, top_k=top_k)
    best = analysis.get_best_config(metric, mode)
    return _clean_config(best)


# ------------------------------------------------------------------
# Internal: search spaces
# ------------------------------------------------------------------

def _build_arch_search_space(search_mode: str) -> dict:
    """
    Build the default architecture search space.
    """
    if search_mode not in ("grid", "random"):
        raise ValueError(f"search_mode must be 'grid' or 'random', got {search_mode!r}")

    depths = [2, 3, 4, 5]
    widths = [512, 1024]
    hidden_dims_options = [[w] * d for w in widths for d in depths]

    lr_options = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    lambda_d_options = [100, 500, 1000, 5000, 10000]

    if search_mode == "grid":
        return {
            "hidden_dims": tune.grid_search(hidden_dims_options),
            "lr":          tune.grid_search(lr_options),
            "lambda_d":    tune.grid_search(lambda_d_options),
            "activation":  "SELU",
            "use_batchnorm": False,
        }
    else:  # random
        return {
            "hidden_dims": tune.choice(hidden_dims_options),
            "lr":          tune.choice(lr_options),
            "lambda_d":    tune.choice(lambda_d_options),
            "activation":  "SELU",
            "use_batchnorm": False,
        }


def _legacy_default_search_space(input_dim: int) -> dict:
    """Search space used by the legacy tune_architecture() function."""
    w = max(64, int(input_dim ** 0.5))
    return {
        "hidden_dims":          tune.grid_search([
            [w * 2] * 2,
            [w * 2] * 4,
            [w * 4] * 4,
            [w * 4] * 6,
        ]),
        "activation":           tune.grid_search(["SELU", "ReLU"]),
        "bottleneck_activation": None,
        "use_batchnorm":        tune.choice([True, False]),
        "dropout_rate":         tune.choice([0.0, 0.1]),
    }


# ------------------------------------------------------------------
# Internal: Ray Tune trainable
# ------------------------------------------------------------------

def _arch_search_trainable(config):
    """
    Ray Tune trainable for architecture search.

    Loads the pre-saved teacher embedding from disk (computed once
    before the sweep), trains a MEDAL student, and lets MEDAL.fit()
    stream metrics to Ray Tune via tune.report().
    """
    import torch.nn as nn
    from medal.model import MEDAL

    X_train    = np.load(Path(config["output_dir"]) / "X_train.npy")
    Z_raw      = np.load(config["_emb_path"])
    normalizer = GlobalEmbeddingNormalizer.load(config["_norm_path"])
    Z_train    = normalizer.transform(Z_raw)

    student = MEDAL(
        input_dim=X_train.shape[1],
        latent_dim=config.get("latent_dim", 2),
        hidden_dims=config.get("hidden_dims", (128, 128)),
        activation=config.get("activation", "SELU"),
        bottleneck_activation=config.get("bottleneck_activation", None),
        final_activation=config.get("final_activation", None),
        lambda_d=config.get("lambda_d", 10),
        lr=config.get("lr", 1e-3),
        epochs=config.get("max_epochs", 3000),
        batch_size=config.get("batch_size", 256),
        warmup=config.get("warmup", 0),
        eta_min=config.get("eta_min", 1e-7),
        use_batchnorm=config.get("use_batchnorm", False),
        dropout_rate=config.get("dropout_rate", 0.0),
        adamw_weight_decay=config.get("adamw_weight_decay", 1e-5),
        factor=config.get("t_factor", 0.9),
        patience=config.get("t_patience", 20),
        criterion=nn.MSELoss,
    )

    student.fit(
        X_train, Z_train,
        verbose=False,
        target_bands=config.get("distill_bands"),
        stability_window=20,
        epsilon_distill=1e-7,
        epsilon_recon=1e-3,
        patience=50,
        return_on_stable=True,
    )


# ------------------------------------------------------------------
# Internal: helpers
# ------------------------------------------------------------------

def _clean_config(config: dict) -> dict:
    """Strip internal / Ray-specific keys from a resolved config dict."""
    return {k: v for k, v in config.items() if k not in _INTERNAL_KEYS}


def _serialisable(obj):
    """Recursively convert non-JSON-serialisable values (e.g. numpy scalars)."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialisable(v) for v in obj]
    if hasattr(obj, "item"):   # numpy scalar
        return obj.item()
    if callable(obj):
        return str(obj)
    return obj


def load_arch_config(output_dir: str | Path) -> dict:
    """
    Load the best architecture + optimisation config from a previous
    :func:`tune_medal_architecture` run.

    Parameters
    ----------
    output_dir : str or Path
        The output_dir used when running tune_medal_architecture.

    Returns
    -------
    dict
        Ready to pass as arch_config to medal.sweep.run_teacher_sweep.

    Example
    -------
    ::

        # Session 1 — architecture search
        medal.tune_medal_architecture(X_train, output_dir="output/arch_search", ...)

        # Session 2 — teacher sweep using saved config
        arch_config = medal.load_arch_config("output/arch_search")
        medal.run_teacher_sweep(X_train, arch_config=arch_config, ...)
    """
    return ArchSearchResults.load(output_dir).best_config


def _print_leaderboard(analysis, metric: str = "distill_loss", top_k: int = 5):
    try:
        df = analysis.results_df.nsmallest(top_k, metric)
        print(f"\nTop {top_k} configs by {metric}:")
        for i, (_, row) in enumerate(df.iterrows(), 1):
            lr_val = row.get("config/lr", "?")
            lr_str = f"{lr_val:.2e}" if isinstance(lr_val, float) else str(lr_val)
            print(
                f"  {i}. {metric}={row[metric]:.4e} | "
                f"hidden_dims={row.get('config/hidden_dims', '?')} | "
                f"lr={lr_str} | "
                f"lambda_d={row.get('config/lambda_d', '?')}"
            )
    except Exception:
        pass
