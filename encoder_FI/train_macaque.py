from __future__ import annotations

import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
import scanpy as sc
import torch
import umap
import ray
from ray import tune
import medal


SMOKE_TEST = os.environ.get("MEDAL_SMOKE_TEST", "1") == "1"

BASE = Path(__file__).resolve().parent
DATA_PATH = (BASE / "../data/macaque IG/m1_train_m1m2_mixbatch_test.h5ad").resolve()
OUTPUT = BASE / ("mac_experiments_smoke" if SMOKE_TEST else "mac_experiments")
CPUS_PER_TRIAL = int(os.environ.get("MEDAL_CPUS_PER_TRIAL", "4"))


def dense_float32(X):
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def PCA_UMAP(X, *, n_components=2, n_pcs=100, n_neighbors=20, random_state=0):
    if n_pcs > min(X.shape):
        raise ValueError(f"n_pcs={n_pcs} exceeds min(X.shape)={min(X.shape)}.")
    if n_neighbors >= len(X):
        raise ValueError("n_neighbors must be smaller than the teacher sample count.")
    X_pca = PCA(n_components=n_pcs, random_state=random_state).fit_transform(X)
    return umap.UMAP(n_components=n_components, random_state=random_state,
                     n_neighbors=n_neighbors, n_jobs=1).fit_transform(X_pca)


def main():
    print("SMOKE_TEST:", SMOKE_TEST, flush=True)
    print("MEDAL source:", medal.__file__, flush=True)
    print("Input:", DATA_PATH, flush=True)

    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    ray_cpus = allocated_cpus - 2

    if CPUS_PER_TRIAL < 1 or ray_cpus < CPUS_PER_TRIAL:
        raise RuntimeError(f"Need at least {CPUS_PER_TRIAL + 2} allocated CPUs.")

    # Smoke tests deliberately use CPU, even on a GPU node.
    gpu_count = 0 if SMOKE_TEST else torch.cuda.device_count()

    if not SMOKE_TEST and gpu_count < 1:
        raise RuntimeError("No CUDA GPU is visible for full training.")

    resources = {"cpu": CPUS_PER_TRIAL, "gpu": 0 if SMOKE_TEST else 1}

    max_parallel = ray_cpus // CPUS_PER_TRIAL
    if not SMOKE_TEST:
        max_parallel = min(max_parallel, gpu_count)

    print(
        f"Ray CPUs={ray_cpus}; GPUs={gpu_count}; "
        f"maximum concurrent trials={max_parallel}",
        flush=True,
    )

    ray_tmp = tempfile.mkdtemp(
        prefix=f"ray_{os.getuid()}_",
        dir="/tmp",
    )

    ray.init(
        address="local",
        _temp_dir=ray_tmp,
        include_dashboard=False,
        num_cpus=ray_cpus,
        num_gpus=gpu_count,
    )

    try:
        print("LOADING DATA", flush=True)
        adata = sc.read_h5ad(DATA_PATH)
        train_idx = np.flatnonzero((adata.obs["split"] == "train").to_numpy())
        test_idx = np.flatnonzero((adata.obs["split"] == "test").to_numpy())

        if not len(train_idx) or not len(test_idx):
            raise ValueError("Train and test splits must be nonempty.")

        # Subset BEFORE converting sparse matrices to dense arrays.
        if SMOKE_TEST:
            rng = np.random.default_rng(0)
            train_idx = rng.choice(
                train_idx,
                size=min(1000, len(train_idx)),
                replace=False,
            )
            test_idx = rng.choice(
                test_idx,
                size=min(200, len(test_idx)),
                replace=False,
            )

        X_train = dense_float32(adata[train_idx].X)
        X_test = dense_float32(adata[test_idx].X)
        del adata

        if not np.isfinite(X_train).all():
            raise ValueError("Training data contains NaN or infinity.")
        if not np.isfinite(X_test).all():
            raise ValueError("Test data contains NaN or infinity.")

        print(
            f"Train={X_train.shape}; test={X_test.shape}",
            flush=True,
        )

        teacher = medal.TeacherSpec(
            name="PCA_UMAP",
            version="1",
            embedding_fn=PCA_UMAP,
            default_params={
                "n_neighbors": 10 if SMOKE_TEST else 20,
                "n_pcs": 20 if SMOKE_TEST else 100,
                "random_state": 0,
            },
            sweep=(
                medal.SweepSpec(
                    param_name="n_neighbors",
                    values=(10, 20),
                )
                if SMOKE_TEST else
                medal.SweepSpec(
                    param_name="n_neighbors",
                    minimum=20,
                    maximum=500,
                    scale="log",
                    n_values=10,
                    dtype="int",
                )
            ),
        )

        search_space = None
        if SMOKE_TEST:
            search_space = {
                "hidden_dims": tune.grid_search([[32], [64]]),
                "activation": "SELU",
                "use_batchnorm": False,
                "dropout_rate": 0.0,
                "lr": 1e-3,
                "lambda_d": 10,
            }

        print("STARTING ARCHITECTURE SEARCH", flush=True)

        result = medal.tune_medal_architecture(
            X_train,
            teacher=teacher,
            latent_dim=2,
            search_mode="random",
            search_space=search_space,
            num_samples=1 if SMOKE_TEST else 20,
            max_epochs=5 if SMOKE_TEST else 3000,
            resources_per_trial=resources,
            output_dir=OUTPUT / "custom_arch",
        )

        arch_config = dict(result.best_config)
        if SMOKE_TEST:
            arch_config["max_epochs"] = 5

        print("STARTING TEACHER SWEEP", flush=True)

        sweep = medal.run_teacher_sweep(
            X_train,
            teacher=teacher,
            latent_dim=2,
            arch_config=arch_config,
            seeds=[0] if SMOKE_TEST else [0, 1, 2],
            resources_per_trial=resources,
            output_dir=OUTPUT / "custom_sweep",
        )

        df = sweep.load_metrics(X_test)
        df.to_csv(OUTPUT / "sweep_metrics.csv", index=False)

        if SMOKE_TEST:
            reloaded = medal.SweepResults.load(
                OUTPUT / "custom_sweep"
            )
            reloaded_df = reloaded.load_metrics(X_test)

            assert len(df) == 6, (
                "Expected two parameters × one seed × three splits."
            )
            assert set(df["split"]) == {"Train", "Val", "Test"}
            assert df[teacher.sweep.param_name].nunique() == 2
            assert np.isfinite(df["recon_loss"]).all()

            train_metrics = df.loc[df["split"] == "Train"]
            assert np.isfinite(train_metrics["distill_mse"]).all()

            pd.testing.assert_frame_equal(
                df.reset_index(drop=True),
                reloaded_df.reset_index(drop=True),
            )

            print(
                "SMOKE TEST PASSED: training, checkpoint loading, "
                "metadata reload, and metrics.",
                flush=True,
            )
            return

        param_name = teacher.sweep.param_name
        opt_param = medal.select_teacher_param(
            df,
            param_col=param_name,
        )
        print(f"Selected {param_name}: {opt_param}", flush=True)

        fig = medal.plot_reconstruction_error(
            df,
            opt_param,
            param_col=param_name,
        )
        fig.savefig(
            OUTPUT / "recon_error.png",
            dpi=200,
            bbox_inches="tight",
        )

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()