"""
Run a MEDAL sweep for a paper dataset.

Two modes:
  teacher_sweep  — sweep the teacher hyperparameter (e.g. n_neighbors) at a
                   fixed latent dimensionality.  Uses TEACHER_SWEEP_SPECS.
  rank_sweep     — sweep the latent dimensionality at a fixed teacher
                   hyperparameter.  Uses RANK_SWEEP_SPECS.

Examples
--------
# MNIST × UMAP teacher sweep, 5 seeds:
python eval/run_sweep.py \\
    --dataset mnist --teacher umap --mode teacher_sweep \\
    --output_dir /share/ctn/users/bnc2119/results/mnist_umap \\
    --seeds 0 1 2 3 4 \\
    --ray_storage_path /tmp/ray_results

# MNIST × UMAP rank sweep (vary latent dim), 5 seeds:
python eval/run_sweep.py \\
    --dataset mnist --teacher umap --mode rank_sweep \\
    --output_dir /share/ctn/users/bnc2119/results/mnist_umap_rank \\
    --seeds 0 1 2 3 4 \\
    --ray_storage_path /tmp/ray_results
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

# Ensure the repo root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import medal
from eval.configs import (
    INIT_CONFIG, DISTILL_BANDS_DICT,
    TEACHER_SWEEP_SPECS, RANK_SWEEP_SPECS,
)
from eval.data import load_and_split

# t_n_neighbors is the legacy key used in the sweep specs;
# medal.run_teacher_sweep expects n_neighbors.
_KEY_MAP = {"t_n_neighbors": "n_neighbors"}

_SPECS = {
    "teacher_sweep": TEACHER_SWEEP_SPECS,
    "rank_sweep":    RANK_SWEEP_SPECS,
}


def _spec_to_param_grid(spec: dict) -> list[dict]:
    """
    Expand a sweep spec (dict of key → list-of-values) into a list of
    tc dicts, translating legacy key names along the way.

    Example
    -------
    {"t_n_neighbors": [5, 15], "min_dist": [0.1], "n_components": [2]}
    →
    [{"n_neighbors": 5,  "min_dist": 0.1, "n_components": 2},
     {"n_neighbors": 15, "min_dist": 0.1, "n_components": 2}]
    """
    translated = {_KEY_MAP.get(k, k): list(v) for k, v in spec.items()}
    keys   = list(translated.keys())
    values = list(translated.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_sweep(
    dataset: str,
    teacher: str,
    mode: str,
    output_dir: str,
    seeds: list[int],
    ray_storage_path: str | None,
    save_checkpoints: bool,
    verbose: bool,
):
    # ── data ────────────────────────────────────────────────────────────
    test_size = INIT_CONFIG[dataset].get("test_size", 0.2)
    X_train, X_test = load_and_split(dataset, test_size=test_size, seed=0)
    print(f"[{dataset}] Train: {X_train.shape}  Test: {X_test.shape}")

    # ── arch + training config ───────────────────────────────────────────
    arch_config = {k: v for k, v in INIT_CONFIG[dataset].items() if k != "test_size"}

    # ── sweep grid ───────────────────────────────────────────────────────
    all_specs = _SPECS[mode]
    dataset_specs = all_specs.get(dataset)
    if dataset_specs is None:
        raise ValueError(
            f"No {mode} entry for dataset {dataset!r}. "
            f"Available datasets: {list(all_specs.keys())}"
        )
    teacher_spec = dataset_specs.get(teacher)
    if teacher_spec is None:
        raise ValueError(
            f"No {mode} spec for teacher {teacher!r} on dataset {dataset!r}. "
            f"Available teachers: {list(dataset_specs.keys())}"
        )
    param_grid = _spec_to_param_grid(teacher_spec)

    print(f"[{dataset}] {mode} / {teacher}: "
          f"{len(param_grid)} configs × {len(seeds)} seeds "
          f"= {len(param_grid) * len(seeds)} trials")

    # ── distillation bands ───────────────────────────────────────────────
    distill_bands = DISTILL_BANDS_DICT.get(dataset, [(1e-12, 9e-6)])

    # ── latent dim ───────────────────────────────────────────────────────
    # For teacher_sweep: n_components is fixed across all tc dicts.
    # For rank_sweep:    n_components varies per tc; _sweep_trainable picks
    #                    it up from tc["n_components"] automatically.
    latent_dim = param_grid[0].get("n_components", 2)

    # ── run ──────────────────────────────────────────────────────────────
    results = medal.run_teacher_sweep(
        X_train,
        output_dir=output_dir,
        teacher=teacher,
        arch_config=arch_config,
        param_grid=param_grid,
        latent_dim=latent_dim,
        seeds=seeds,
        distill_bands=distill_bands,
        ray_storage_path=ray_storage_path,
        save_checkpoints=save_checkpoints,
        verbose=verbose,
        resources_per_trial = {"cpu": 4, "gpu": 1}
    )

    print(f"\nSweep complete. Results saved to: {results.output_dir}")
    print(f"To load later:\n  results = medal.SweepResults.load({str(output_dir)!r})")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run a MEDAL sweep (teacher or rank).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        choices=list(INIT_CONFIG.keys()),
                        help="Dataset name.")
    parser.add_argument("--teacher", required=True,
                        choices=["umap", "tsne", "spectral", "phate", "pca"],
                        help="Teacher algorithm.")
    parser.add_argument("--mode", default="teacher_sweep",
                        choices=["teacher_sweep", "rank_sweep"],
                        help=(
                            "teacher_sweep: vary teacher hyperparameter at fixed latent dim. "
                            "rank_sweep: vary latent dimensionality at fixed teacher hyperparameter."
                        ))
    parser.add_argument("--output_dir", required=True,
                        help="Directory for embeddings, checkpoints, and metadata.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="AE random seeds.")
    parser.add_argument("--ray_storage_path", default=None,
                        help="Ray trial log directory. Use /tmp/... to avoid filling shared storage.")
    parser.add_argument("--no_save_checkpoints", action="store_true",
                        help="Skip saving model checkpoints (only sweep_summary.csv is written).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_sweep(
        dataset=args.dataset,
        teacher=args.teacher,
        mode=args.mode,
        output_dir=args.output_dir,
        seeds=args.seeds,
        ray_storage_path=args.ray_storage_path,
        save_checkpoints=not args.no_save_checkpoints,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
