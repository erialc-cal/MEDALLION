"""
Centralised path helpers for teacher embeddings and student checkpoints.

All paths are relative to a caller-supplied base_dir so there are
no hard-coded filesystem assumptions in the library.
"""
from pathlib import Path
from medal.custom_teacher import is_custom_key, stable_digest

# Maps teacher name to the name of its primary hyperparameter (the one
# that is swept in a teacher-hyperparameter sweep).
TEACHER_PARAM_KEY = {
    "umap":     "n_neighbors",
    "tsne":     "perplexity",
    "spectral": "n_neighbors",
    "phate":    "n_neighbors",
    "isomap":   "n_neighbors",
    "pca":      "n_components",
}


def _teacher_suffix(teacher: str, tc: dict) -> str:
    """Build the filename suffix that encodes the teacher hyperparameters."""
    n = tc.get("n_components", 2)
    if is_custom_key(teacher):
        return f"{teacher}_{stable_digest(tc)[:20]}"
    if teacher == "umap":
        return (
            f"umap_{tc['n_neighbors']}_{tc['min_dist']}" if n == 2
            else f"umap{n}_{tc['n_neighbors']}_{tc['min_dist']}"
        )
    if teacher == "tsne":
        return (
            f"tsne_{tc['perplexity']}" if n == 2
            else f"tsne{n}_{tc['perplexity']}"
        )
    if teacher == "pca":
        return f"pca{n}"
    if teacher == "isomap":
        return f"isomap_{tc['n_neighbors']}"
    if teacher == "spectral":
        return f"spectral{n}_{tc['n_neighbors']}"
    if teacher == "phate":
        return f"phate{n}_{tc['n_neighbors']}"

    raise ValueError(f"Unknown teacher: {teacher!r}")


def teacher_embedding_path(base_dir, dataset_name: str, teacher: str, tc: dict) -> Path:
    """
    Return the .npy path for a cached teacher embedding.

    Teacher embeddings are keyed by (teacher, hyperparameters) only — not by
    AE seed, because a single embedding is shared across all AE seeds for a
    given hyperparameter configuration.

    Parameters
    ----------
    base_dir : str or Path
        Root output directory for the experiment.
    dataset_name : str
        Short name used in the filename (e.g. "mnist", "hydra").
    teacher : str
        Teacher algorithm name.
    tc : dict
        Teacher hyperparameter dict (e.g. {"n_neighbors": 18, "min_dist": 0.1}).
    """
    suffix = _teacher_suffix(teacher, tc)
    return Path(base_dir) / "embeddings" / f"{dataset_name}_{suffix}_train.npy"


def teacher_norm_path(embedding_path) -> Path:
    """Return the normaliser .pkl path that accompanies an embedding file."""
    return Path(embedding_path).with_suffix(".norm.pkl")


def student_ckpt_path(base_dir, prefix: str) -> Path:
    """Return the final checkpoint path for a trained student model."""
    return Path(base_dir) / f"{prefix}_ckpts" / "final.pt"
