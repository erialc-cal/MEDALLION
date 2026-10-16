"""
Model I/O utilities: loading checkpoints, running inference, computing losses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from medal.model import AutoEncoder
from medal.normalizer import GlobalEmbeddingNormalizer


def load_model(
    ckpt_path: str | Path,
    input_dim: int,
    hidden_dims: tuple,
    latent_dim: int = 2,
    activation: str = "SELU",
    use_batchnorm: bool = False,
    bottleneck_activation=None,
    final_activation=None,
    dropout_rate: float = 0.0,
) -> AutoEncoder:
    """
    Load a trained medal.model.AutoEncoder from a checkpoint file.

    Parameters
    ----------
    ckpt_path : str or Path
        Path to the .pt checkpoint saved by :meth:`MEDAL.fit`.
    input_dim : int
        Must match the value used during training.
    hidden_dims : tuple of int
        Must match the value used during training.
    latent_dim : int
        Must match the value used during training.
    activation : str
        Activation class name (e.g. "SELU").
    use_batchnorm : bool
        Must match the value used during training.

    Returns
    -------
    model : AutoEncoder
        Loaded model in eval mode on CPU.
    """
    import torch.nn as nn
    act = getattr(nn, activation) if isinstance(activation, str) else activation
    bn_act = (
        getattr(nn, bottleneck_activation)
        if isinstance(bottleneck_activation, str) else bottleneck_activation
    )
    fin_act = (
        getattr(nn, final_activation)
        if isinstance(final_activation, str) else final_activation
    )

    model = AutoEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=act,
        bottleneck_activation=bn_act,
        final_activation=fin_act,
        use_batchnorm=use_batchnorm,
        dropout_rate=dropout_rate,
    )

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # unwrap checkpoint wrapper if present
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # strip DDP prefix if any
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        import warnings
        warnings.warn(
            f"load_model: missing={len(missing)} keys, unexpected={len(unexpected)} keys. "
            "Check that the architecture arguments match the checkpoint."
        )

    model.eval()
    return model


def embed(
    model: AutoEncoder,
    X: np.ndarray,
    normalizer: Optional[GlobalEmbeddingNormalizer] = None,
    device: Optional[str] = None,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Encode *X* through the autoencoder bottleneck.

    Parameters
    ----------
    model : AutoEncoder
        Trained model (from load_model or medal.MEDAL.model).
    X : array-like of shape (n_samples, input_dim)
    normalizer : GlobalEmbeddingNormalizer, optional
        If given, apply normalizer.inverse_transform to the raw latent
        codes so they are back in the original teacher's coordinate frame.
    device : str, optional
        Torch device string.  Defaults to the device the model is already on.
    batch_size : int
        Process this many samples at a time to avoid OOM on large datasets.

    Returns
    -------
    Z : np.ndarray of shape (n_samples, latent_dim)
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    X = np.asarray(X, dtype=np.float32)
    model = model.to(device)
    model.eval()

    chunks = []
    for start in range(0, len(X), batch_size):
        x_chunk = torch.tensor(X[start:start + batch_size]).to(device)
        with torch.no_grad():
            _, z = model(x_chunk)
        chunks.append(z.cpu().numpy())

    Z = np.concatenate(chunks, axis=0)
    if normalizer is not None:
        Z = normalizer.inverse_transform(Z)
    return Z


def compute_losses(
    model: AutoEncoder,
    X: np.ndarray,
    teacher_z: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> tuple[float, Optional[float]]:
    """
    Compute reconstruction MSE and (optionally) distillation MSE.

    Returns
    -------
    recon_mse : float
    distill_mse : float or None
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    model.eval()
    X_tensor = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    with torch.no_grad():
        x_recon, student_z = model(X_tensor)

    recon_mse = mean_squared_error(X, x_recon.cpu().numpy())
    if teacher_z is not None:
        distill_mse = mean_squared_error(teacher_z, student_z.cpu().numpy())
        return recon_mse, distill_mse
    return recon_mse, None


def eval_student(student, X: np.ndarray, Z: np.ndarray) -> dict:
    """Convenience wrapper: returns a dict with recon_mse and distill_mse."""
    rmse, dmse = compute_losses(model=student.model, X=X, teacher_z=Z)
    return {"recon_mse": rmse, "distill_mse": dmse}


def embed_with_reconstruction(
    model: AutoEncoder,
    X: np.ndarray,
    device: Optional[str] = None,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode *X* and compute per-point reconstruction error in a single pass.

    Parameters
    ----------
    model : AutoEncoder
    X : array-like of shape (n_samples, input_dim)
    device : str, optional
    batch_size : int

    Returns
    -------
    Z : np.ndarray of shape (n_samples, latent_dim)
    recon_error : np.ndarray of shape (n_samples,)
        Per-sample mean squared reconstruction error.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    X = np.asarray(X, dtype=np.float32)
    model = model.to(device)
    model.eval()

    Z_chunks, recon_chunks = [], []
    for start in range(0, len(X), batch_size):
        x_chunk = torch.tensor(X[start:start + batch_size]).to(device)
        with torch.no_grad():
            x_recon, z = model(x_chunk)
        Z_chunks.append(z.cpu().numpy())
        recon_chunks.append(x_recon.cpu().numpy())

    Z = np.concatenate(Z_chunks, axis=0)
    X_recon = np.concatenate(recon_chunks, axis=0)
    recon_error = np.mean((X - X_recon) ** 2, axis=1)
    return Z, recon_error
