"""
Width analysis: effect of per-layer neuron count on distillation and
reconstruction loss.

Grid-searches over::

    hidden_dims ∈ { [32]*3, [128]*3, [512]*3, [1024]*3 }

repeated 20 times (different AE random seeds) → 80 trials per dataset.
All architectural and optimisation hyperparameters are taken from
eval/configs.py (INIT_CONFIG) and held fixed; only hidden_dims varies.
No early stopping (target_bands=None) and no checkpoints are saved.

MEDAL.fit() calls tune.report() internally at each report_interval
(including the final epoch), so Ray captures distill_loss / recon_loss
automatically.

Output
------
{dataset_key}_depth_analysis.csv  — one row per trial, columns:
    hidden_dims, seed, distill_loss, recon_loss

Example
-------
python eval/run_width_analysis.py \\
    --dataset mnist \\
    --output_dir /share/ctn/users/bnc2119/results/width_analysis/mnist \\
    --ray_storage_path /tmp/ray_width_mnist
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ray import tune

from medal.model import MEDAL
from medal.sweep import _precompute_one_embedding
from medal.normalizer import GlobalEmbeddingNormalizer
from medal._paths import teacher_embedding_path, teacher_norm_path

from eval.configs import INIT_CONFIG
from eval.data import load_and_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIDDEN_DIMS_GRID = [
    [32,   32,   32],
    [128,  128,  128],
    [512,  512,  512],
    [1024, 1024, 1024],
]

N_SEEDS = 10

# Fixed teacher config per algorithm (held constant — we are studying width,
# not teacher sensitivity).
_FIXED_TEACHER_CONFIG = {
    "umap": {"n_neighbors": 15, "min_dist": 0.1, "n_components": 2},
    "tsne": {"perplexity": 30, "n_components": 2},
    "pca":  {"n_components": 2},
}

# "pancan" is an alias for the "gene_cancer" key in INIT_CONFIG / data.py.
_DATASET_ALIAS = {"pancan": "gene_cancer"}


# ---------------------------------------------------------------------------
# Ray Tune trainable
# ---------------------------------------------------------------------------

def _width_trainable(config):
    """
    Ray Tune trainable for one (hidden_dims, seed) trial.

    * target_bands=None  → runs all max_epochs, no early stopping.
    * save_dir=None      → no checkpoint files written to disk.
    * MEDAL.fit() reports distill_loss / recon_loss to Ray at each
      report_interval epoch (including the last), so no explicit
      tune.report() call is needed here.
    """
    output_dir = Path(config["output_dir"])
    teacher    = config["teacher"]
    tc         = config["teacher_config"]
    seed       = config["seed"]
    norm       = config.get("normalize_teacher", True)

    X_train = np.load(output_dir / "X_train.npy")

    emb_path = teacher_embedding_path(output_dir, "data", teacher, tc)
    Z_raw    = np.load(emb_path)
    if norm:
        normalizer = GlobalEmbeddingNormalizer.load(teacher_norm_path(emb_path))
        Z_train    = normalizer.transform(Z_raw)
    else:
        Z_train = Z_raw

    torch.manual_seed(seed)

    student = MEDAL(
        input_dim            = X_train.shape[1],
        latent_dim           = tc.get("n_components", 2),
        hidden_dims          = config["hidden_dims"],
        activation           = config.get("activation", "SELU"),
        bottleneck_activation= config.get("bottleneck_activation", None),
        final_activation     = config.get("final_activation", None),
        lambda_d             = config.get("lambda_d", 100),
        lr                   = config.get("lr", 1e-3),
        epochs               = config.get("max_epochs", 5000),
        batch_size           = config.get("batch_size", 256),
        warmup               = config.get("warmup", 0),
        eta_min              = config.get("eta_min", 1e-7),
        use_batchnorm        = config.get("use_batchnorm", False),
        dropout_rate         = config.get("dropout_rate", 0.0),
        adamw_weight_decay   = config.get("adamw_weight_decay", 1e-5),
        factor               = config.get("t_factor", 0.95),
        patience             = config.get("t_patience", 20),
        criterion            = nn.MSELoss,
    )

    student.fit(
        X_train,
        Z_train,
        target_bands = None,   # no early stopping
        save_dir     = None,   # no checkpoint files
        verbose      = False,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_width_analysis(
    dataset: str,
    teacher: str,
    output_dir: str,
    ray_storage_path: str | None,
) -> None:
    dataset_key = _DATASET_ALIAS.get(dataset.lower(), dataset.lower())

    if dataset_key not in INIT_CONFIG:
        raise ValueError(
            f"Unknown dataset {dataset!r}. "
            f"Available: {list(INIT_CONFIG.keys()) + list(_DATASET_ALIAS)}"
        )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── data ────────────────────────────────────────────────────────────────
    test_size = INIT_CONFIG[dataset_key].get("test_size", 0.2)
    X_train, _X_test = load_and_split(dataset_key, test_size=test_size, seed=0)
    print(f"[{dataset_key}] Train: {X_train.shape}")

    np.save(output_dir / "X_train.npy", X_train)

    # ── teacher embedding (one, shared across all 80 trials) ────────────────
    tc = _FIXED_TEACHER_CONFIG[teacher]
    _precompute_one_embedding(
        output_dir, "data", teacher, tc, X_train,
        normalize=True, verbose=True,
    )

    # ── arch/opt params from INIT_CONFIG — everything fixed except hidden_dims
    arch_cfg = {
        k: v for k, v in INIT_CONFIG[dataset_key].items()
        if k not in ("test_size", "hidden_dims")
    }

    # ── Ray Tune config ──────────────────────────────────────────────────────
    base_cfg = {
        **arch_cfg,
        "hidden_dims":      tune.grid_search(HIDDEN_DIMS_GRID),
        "seed":             tune.grid_search(list(range(N_SEEDS))),
        "output_dir":       str(output_dir),
        "teacher":          teacher,
        "teacher_config":   tc,
        "normalize_teacher": True,
    }

    storage = ray_storage_path or str(output_dir / "ray_results")

    print(
        f"\n[{dataset_key}] Launching {len(HIDDEN_DIMS_GRID)} hidden_dims configs "
        f"× {N_SEEDS} seeds = {len(HIDDEN_DIMS_GRID) * N_SEEDS} trials "
        f"(1 GPU + 4 CPU each)\n"
    )

    analysis = tune.run(
        _width_trainable,
        name=f"width_analysis_{dataset_key}",
        num_samples=1,                          # grid_search covers all combos
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=base_cfg,
        metric="distill_loss",
        mode="min",
        verbose=1,
        max_failures=3,
        storage_path=storage,
    )

    # ── save results CSV ─────────────────────────────────────────────────────
    df = analysis.results_df.copy()

    # Ray Tune may expose config columns as "config/key" or nested under "config"
    col_map = {}
    for raw in ("config/hidden_dims", "config/seed"):
        if raw in df.columns:
            col_map[raw] = raw.split("/")[1]
    df.rename(columns=col_map, inplace=True)

    # Fallback: unpack from the "config" dict column if present
    if "hidden_dims" not in df.columns and "config" in df.columns:
        df["hidden_dims"] = df["config"].apply(lambda c: c.get("hidden_dims"))
        df["seed"]        = df["config"].apply(lambda c: c.get("seed"))

    keep = [c for c in ("hidden_dims", "seed", "distill_loss", "recon_loss")
            if c in df.columns]
    df = df[keep].copy()

    # Represent hidden_dims as a readable string, e.g. "512x3"
    df["hidden_dims"] = df["hidden_dims"].apply(
        lambda h: f"{h[0]}x{len(h)}" if isinstance(h, (list, tuple)) else str(h)
    )

    csv_path = Path(f"{dataset_key}_width_analysis_{teacher}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDone. Results saved to: {csv_path.resolve()}")
    print(df.groupby("hidden_dims")[["distill_loss", "recon_loss"]].mean().to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "MEDAL width analysis: sweep hidden_dims over "
            f"{N_SEEDS} seeds per config ({len(HIDDEN_DIMS_GRID) * N_SEEDS} trials total)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["mnist", "darmanis", "gene_cancer", "pancan"],
        help="Dataset to run on. 'pancan' maps to 'gene_cancer'.",
    )
    parser.add_argument(
        "--teacher", default="umap",
        choices=list(_FIXED_TEACHER_CONFIG.keys()),
        help="Teacher algorithm (config is fixed; we are studying architecture width).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory for the teacher embedding cache and Ray Tune logs.",
    )
    parser.add_argument(
        "--ray_storage_path", default=None,
        help=(
            "Ray trial log directory. "
            "Tip: use /tmp/... to avoid filling shared storage."
        ),
    )
    args = parser.parse_args()

    run_width_analysis(
        dataset          = args.dataset,
        teacher          = args.teacher,
        output_dir       = args.output_dir,
        ray_storage_path = args.ray_storage_path,
    )


if __name__ == "__main__":
    main()
