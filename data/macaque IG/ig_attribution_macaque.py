"""
    PYTHONPATH=src:. python eval/ig_attribution_macaque.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medal.io import load_model


ROOT = Path(__file__).resolve().parents[1]
DRD_MACAQUE = Path("...")
CKPT_PATH = DRD_MACAQUE / "medal_umap2_6_seed0_ckpts" / "final.pt"
OUT_DIR = ROOT / "output" / "ig_attribution_macaque"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIM   = 100
HIDDEN_DIMS = (512, 512, 512, 512)
LATENT_DIM  = 2

N_SAMPLES       = None   # experiments 1 & 3 (None = use all test samples)
N_SAMPLES_EXP2  = None  
IG_STEPS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_everything():
    model = load_model(
        ckpt_path=CKPT_PATH,
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        activation="SELU",
        use_batchnorm=True,
        dropout_rate=0.1,
    )
    model = model.to(DEVICE)
    model.eval()

    X_test = np.load(DRD_MACAQUE / "X_val.npy")
    return model, X_test
    
def sanity_checks(model, X_test):
    print("=== Sanity Checks ===")
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  Device: {DEVICE}")
    print(f"  X_test shape: {X_test.shape}")

    x = torch.tensor(X_test[:10], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        x_recon, z = model(x)

    assert z.shape == (10, LATENT_DIM), f"Encoder output should be (10, {LATENT_DIM}), got {z.shape}"
    assert x_recon.shape == (10, INPUT_DIM), f"Recon shape should be (10, {INPUT_DIM}), got {x_recon.shape}"
    recon_mse = ((x - x_recon) ** 2).mean().item()
    print(f"  Encoder output dim: {z.shape[1]}")
    print(f"  Sample recon MSE (10 samples): {recon_mse:.6f}")
    print(f"  z range: [{z.min().item():.3f}, {z.max().item():.3f}]")
    print(f"  x_recon range: [{x_recon.min().item():.4f}, {x_recon.max().item():.4f}]")
    print()

class _Encoder(nn.Module):
    """x -> encoder latent (batch, 2)."""
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
    def forward(self, x):
        return self.encoder(x)


class _Autoencoder(nn.Module):
    """x -> reconstruction (batch, INPUT_DIM)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)[0]


class _ReconLoss(nn.Module):
    """x -> per-sample reconstruction MSE (batch,)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        x_recon = self.model(x)[0]
        return ((x - x_recon) ** 2).mean(dim=1)


# ---------------------------------------------------------------------------
# Experiment 1: Encoder IG
# ---------------------------------------------------------------------------

def run_encoder_ig(model, X_test):
    print("=== Experiment 1: Encoder IG ===")
    t0 = time.time()

    ig = IntegratedGradients(_Encoder(model))

    n = N_SAMPLES if N_SAMPLES is not None else len(X_test)
    if n < len(X_test):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_test), size=n, replace=False)
        x_batch = torch.tensor(X_test[idx], dtype=torch.float32, device=DEVICE)
    else:
        x_batch = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    baseline = torch.zeros_like(x_batch)

    attr_z1 = ig.attribute(x_batch, baselines=baseline, target=0,
                           n_steps=IG_STEPS,
                           internal_batch_size=len(x_batch))
    attr_z2 = ig.attribute(x_batch, baselines=baseline, target=1,
                           n_steps=IG_STEPS,
                           internal_batch_size=len(x_batch))

    importance = (attr_z1.abs() + attr_z2.abs()).cpu().numpy()  # (N, 100)
    avg_map = importance.mean(axis=0)  # (100,)
    print(f"  Done in {time.time()-t0:.1f}s  ({len(x_batch)} samples)")
    assert np.all(np.isfinite(avg_map)), "Non-finite attribution values!"
    return avg_map


# ---------------------------------------------------------------------------
# Experiment 2: Whole-autoencoder IG
# ---------------------------------------------------------------------------

def run_autoencoder_ig(model, X_test):
    print("=== Experiment 2: Whole-Autoencoder IG ===")
    t0 = time.time()

    ig = IntegratedGradients(_Autoencoder(model))

    n = N_SAMPLES_EXP2 if N_SAMPLES_EXP2 is not None else len(X_test)
    if n < len(X_test):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_test), size=n, replace=False)
        x_batch = torch.tensor(X_test[idx], dtype=torch.float32, device=DEVICE)
    else:
        x_batch = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    baseline = torch.zeros_like(x_batch)

    total = torch.zeros_like(x_batch)
    for j in range(INPUT_DIM):
        attr_j = ig.attribute(x_batch, baselines=baseline, target=j,
                              n_steps=IG_STEPS,
                              internal_batch_size=len(x_batch))
        total += attr_j.abs()
        if (j + 1) % 20 == 0:
            print(f"  {j+1}/{INPUT_DIM} output dims")

    avg_map = total.cpu().numpy().mean(axis=0)  # (100,)
    print(f"  Done in {time.time()-t0:.1f}s  ({len(x_batch)} samples)")
    assert np.all(np.isfinite(avg_map)), "Non-finite attribution values!"
    return avg_map


# ---------------------------------------------------------------------------
# Experiment 3: Reconstruction-loss IG
# ---------------------------------------------------------------------------

def run_recon_loss_ig(model, X_test):
    print("=== Experiment 3: Reconstruction-Loss IG ===")
    t0 = time.time()

    ig = IntegratedGradients(_ReconLoss(model))

    n = N_SAMPLES if N_SAMPLES is not None else len(X_test)
    if n < len(X_test):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_test), size=n, replace=False)
        x_batch = torch.tensor(X_test[idx], dtype=torch.float32, device=DEVICE)
    else:
        x_batch = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    baseline = torch.zeros_like(x_batch)

    attr = ig.attribute(x_batch, baselines=baseline,
                        n_steps=IG_STEPS,
                        internal_batch_size=len(x_batch))

    attr_np = attr.cpu().numpy()  # (N, 100)
    avg_signed = attr_np.mean(axis=0)
    avg_abs = np.abs(attr_np).mean(axis=0)
    print(f"  Done in {time.time()-t0:.1f}s  ({len(x_batch)} samples)")
    assert np.all(np.isfinite(avg_abs)), "Non-finite attribution values!"
    return avg_abs, avg_signed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    model, X_test = load_everything()
    sanity_checks(model, X_test)

    # --- Experiment 1: Encoder IG ---
    enc_map = run_encoder_ig(model, X_test)

    # --- Experiment 2: Whole-autoencoder IG ---
    ae_map = run_autoencoder_ig(model, X_test)

    # --- Experiment 3: Reconstruction-loss IG ---
    recon_map_abs, recon_map_signed = run_recon_loss_ig(model, X_test)

    # --- Save raw data ---
    np.savez(
        OUT_DIR / "ig_attribution_data.npz",
        encoder_ig=enc_map,
        autoencoder_ig=ae_map,
        recon_loss_ig_abs=recon_map_abs,
        recon_loss_ig_signed=recon_map_signed,
    )
    print(f"\n  Raw data -> {OUT_DIR / 'ig_attribution_data.npz'}")

  
if __name__ == "__main__":
    main()
