"""
MEDAL — Manifold Embedding Distillation via Autoencoder Learning.

Quick-start::

    import medal

    # 1. Find the best autoencoder architecture for your data
    analysis = medal.tune_architecture(
        X_train,
        output_dir="experiments/arch",
        teacher="tsne",
        teacher_params={"perplexity": 30},
    )
    arch_config = medal.get_best_config(analysis)

    # 2. Sweep teacher hyperparameters and train one student per setting
    results = medal.run_teacher_sweep(
        X_train,
        output_dir="experiments/sweep",
        teacher="tsne",
        arch_config=arch_config,
    )

    # 3. Select the optimal hyperparameter value
    df = results.load_metrics(X_test)
    opt = medal.select_teacher_param(df, param_col="perplexity")

    # 4. Visualise
    medal.plot_reconstruction_error(df, opt, param_col="perplexity")
"""
from medal.model import MEDAL, AutoEncoder
from medal.normalizer import GlobalEmbeddingNormalizer
from medal.tuning import tune_medal_architecture, ArchSearchResults, tune_architecture, get_best_config, load_arch_config
from medal.sweep import run_teacher_sweep, SweepResults
from medal.selection import (
    select_teacher_param,
    plot_reconstruction_error,
    plot_distortion_map,
)
from medal.teacher import get_teacher_embeddings, build_param_grid
from medal.io import load_model, embed, compute_losses, eval_student
from medal.custom_teacher import TeacherSpec, SweepSpec
__all__ = [
    # Core model
    "MEDAL",
    "AutoEncoder",
    "GlobalEmbeddingNormalizer",
    # Architecture search
    "tune_medal_architecture",
    "ArchSearchResults",
    "tune_architecture",
    "get_best_config",
    "load_arch_config",
    # Teacher sweep
    "run_teacher_sweep",
    "SweepResults",
    # Selection & plotting
    "select_teacher_param",
    "plot_reconstruction_error",
    "plot_distortion_map",
    # Lower-level utilities
    "get_teacher_embeddings",
    "build_param_grid",
    "load_model",
    "embed",
    "compute_losses",
    "eval_student",
    # Custom teacher support
    "TeacherSpec",
    "SweepSpec",
]
