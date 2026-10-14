from pathlib import Path
from copy import deepcopy
from itertools import combinations
import json

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from medal.io import load_model

N_EXP = 5
IG_STEPS = 2000
TOP_K = 10
teacher_error_threshold = 1e-4

BASE = Path("/Users/clairehe/Documents/GitHub/MEDAL_FI")
OUT = BASE / "data/baby_mnist_medal"
SOURCE = BASE / "encoder_FI/experiments/arch_search"
SAVE = SOURCE / "comp_files/seed_consistency"
SAVE.mkdir(parents=True, exist_ok=True)

cfg = json.loads((OUT / "config.json").read_text())
X = torch.from_numpy(
    np.load(SOURCE / "X_train.npy").astype(np.float32)
)

# Fix the same teacher and normalization for every run.
normalization = np.load(OUT / "teacher_normalization.npz")
raw_Z = np.load(SOURCE / "embeddings/arch_search_teacher.npy")
Z = torch.from_numpy(
    ((raw_Z - normalization["mean"]) / normalization["scale"])
    .astype(np.float32)
)

template = load_model(
    OUT / "baby_mnist_ckpts/final.pt",
    input_dim=64,
    hidden_dims=tuple(cfg["hidden_dims"]),
    latent_dim=2,
    activation=cfg["activation"],
    use_batchnorm=cfg["use_batchnorm"],
    dropout_rate=cfg["dropout_rate"],
).cpu()

torch.set_num_threads(4)

def fresh_model(seed):
    torch.manual_seed(seed)
    model = deepcopy(template)
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return model

def train_one(initial_model, objective, seed):
    model = deepcopy(initial_model)
    network = model.encoder if objective == "encoder_only" else model

    # Same minibatch order for both objectives within a seed.
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Z),
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["adamw_weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=cfg["t_factor"],
        patience=cfg["t_patience"],
        min_lr=cfg["eta_min"],
    )

    for epoch in range(cfg["max_epochs"]):
        network.train()
        for xb, zb in loader:
            optimizer.zero_grad()

            if objective == "medal":
                reconstruction, embedding = model(xb)
                reconstruction_loss = (reconstruction - xb).square().mean()
            else:
                embedding = model.encoder(xb)
                reconstruction_loss = 0.0

            distillation_loss = (embedding - zb).square().mean()

            # Same distillation weighting in both objectives:
            # reconstruction is the experimental difference.
            loss = cfg["lambda_d"] * distillation_loss + reconstruction_loss
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            mse = (model.encoder(X) - Z).square().mean().item()
        scheduler.step(mse)

        if (epoch + 1) % 500 == 0:
            print(seed, objective, epoch + 1, f"teacher MSE={mse:.3g}")

    # Same final-epoch selection rule for both objectives.
    model.eval()
    return model

def compute_ig(encoder, inputs, n_steps=2000):
    ig = IntegratedGradients(encoder)
    attrs, errors = [], []

    for coordinate in range(Z.shape[1]):
        coordinate_attrs, coordinate_errors = [], []
        for batch in inputs.split(32):
            a, delta = ig.attribute(
                batch,
                baselines=torch.zeros_like(batch),
                target=coordinate,
                n_steps=n_steps,
                internal_batch_size=128,
                return_convergence_delta=True,
            )
            coordinate_attrs.append(a.detach().numpy())
            coordinate_errors.append(delta.detach().numpy())

        attrs.append(np.concatenate(coordinate_attrs))
        errors.append(np.concatenate(coordinate_errors))

    return np.stack(attrs, axis=1), np.stack(errors, axis=1)

# At least two seeds are needed for within-objective consistency.
seeds = range(N_EXP)
runs = {"encoder_only": {}, "medal": {}}

# Save settings and image/teacher references before expensive computation.
(SAVE / "settings.json").write_text(json.dumps({
    "seeds": list(seeds),
    "config": cfg,
    "ig_steps": IG_STEPS,
    "ig_baseline": "zero",
    "top_k": TOP_K,
    "teacher_error_threshold": teacher_error_threshold,
    "evaluation_split": "train",
    "checkpoint_selection": "final_epoch",
    "loss": "lambda_d * distillation_mse + reconstruction_mse (MEDAL only)",
}, indent=2))
np.savez_compressed(
    SAVE / "reference.npz",
    X=X.numpy(), teacher_embedding=Z.numpy(),
    teacher_mean=normalization["mean"], teacher_scale=normalization["scale"],
)

for seed in seeds:
    initial_model = fresh_model(seed)

    for objective in runs:
        model = train_one(initial_model, objective, seed)
        with torch.no_grad():
            predictions = model.encoder(X).numpy()

        # Preserve trained weights even if attribution is interrupted.
        torch.save(model.state_dict(), SAVE / f"{objective}_seed_{seed}.pt")
        attrs, delta = compute_ig(model.encoder, X, n_steps=IG_STEPS)
        runs[objective][seed] = {
            "predictions": predictions,
            "teacher_error": np.mean(
                (predictions - Z.numpy()) ** 2, axis=1
            ),
            "ig": attrs,
            "delta": delta,
        }

        print(
            seed, objective,
            "teacher MSE:", runs[objective][seed]["teacher_error"].mean(),
            "mean |IG delta|:", np.abs(delta).mean(),
        )
        np.savez_compressed(
            SAVE / f"{objective}_seed_{seed}.npz",
            **runs[objective][seed],
        )
        print(f"Saved {objective}, seed {seed}, to {SAVE}")

def cosine_per_image(a, b):
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        (a * b).sum(axis=1),
        denominator,
        out=np.full(len(a), np.nan),
        where=denominator > 0,
    )

def explanation_metrics(A, B, k=TOP_K):
    if A.shape != B.shape or A.ndim != 3:
        raise ValueError("IG arrays must have the same (images, coordinates, pixels) shape")
    if not 1 <= k <= A.shape[-1]:
        raise ValueError("k must be between 1 and the number of pixels")
    signed = cosine_per_image(
        A.reshape(len(A), -1),
        B.reshape(len(B), -1),
    )
    a = np.linalg.norm(A, axis=1)
    b = np.linalg.norm(B, axis=1)
    magnitude = cosine_per_image(a, b)

    a_top = np.argsort(-a, axis=1, kind="stable")[:, :k]
    b_top = np.argsort(-b, axis=1, kind="stable")[:, :k]
    overlap = np.array([
        np.intersect1d(x, y).size / k
        for x, y in zip(a_top, b_top)
    ])
    overlap[(a.max(axis=1) == 0) | (b.max(axis=1) == 0)] = np.nan
    return signed, magnitude, overlap

# Illustrative threshold in normalized teacher-coordinate MSE.
# Choose before examining explanation scores, and sweep it afterward.
rows = []

for seed_a, seed_b in combinations(seeds, 2):
    # Use the SAME eligible images for both objectives:
    # all four models must match the teacher within the threshold.
    eligible = np.logical_and.reduce([
        runs[objective][seed]["teacher_error"] <= teacher_error_threshold
        for objective in runs
        for seed in (seed_a, seed_b)
    ])

    for objective in runs:
        a = runs[objective][seed_a]
        b = runs[objective][seed_b]
        signed, magnitude, overlap = explanation_metrics(a["ig"], b["ig"])

        def selected_mean(values):
            valid = eligible & np.isfinite(values)
            return values[valid].mean() if valid.any() else np.nan

        rows.append({
            "objective": objective,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "n_images": int(eligible.sum()),
            "teacher_mse_a": selected_mean(a["teacher_error"]),
            "teacher_mse_b": selected_mean(b["teacher_error"]),
            "student_pair_mse": selected_mean(
                np.mean(
                    (a["predictions"] - b["predictions"]) ** 2,
                    axis=1,
                )
            ),
            "signed_cosine": selected_mean(signed),
            "magnitude_cosine": selected_mean(magnitude),
            f"top{TOP_K}_overlap": selected_mean(overlap),
        })

# Explicit columns allow a one-seed run to save a valid empty summary.
metrics = ["signed_cosine", "magnitude_cosine", f"top{TOP_K}_overlap"]
columns = [
    "objective", "seed_a", "seed_b", "n_images", "teacher_mse_a",
    "teacher_mse_b", "student_pair_mse", *metrics,
]
pair_results = pd.DataFrame(rows, columns=columns)
pair_results.to_csv(SAVE / "pair_results.csv", index=False)

if pair_results.empty:
    differences = pd.DataFrame(
        columns=metrics,
        index=pd.MultiIndex.from_tuples([], names=["seed_a", "seed_b"]),
    )
    summary = pd.DataFrame(columns=metrics)
    summary.index.name = "objective"
    print("Runs saved. Set N_EXP >= 2 to calculate across-seed consistency.")
else:
    print(pair_results.to_string(index=False))
    summary = pair_results.groupby("objective")[metrics].mean()
    print("\nMean within-objective agreement:")
    print(summary)

    # Seed pairs share models; these are descriptive paired differences.
    paired = pair_results.pivot(
        index=["seed_a", "seed_b"], columns="objective", values=metrics,
    )
    differences = pd.DataFrame({
        metric: paired[(metric, "medal")] - paired[(metric, "encoder_only")]
        for metric in metrics
    })
    print("\nMEDAL minus encoder-only agreement:")
    print(differences)
    print("\nMean paired difference:")
    print(differences.mean())

summary.to_csv(SAVE / "agreement_summary.csv")
differences.to_csv(SAVE / "paired_differences.csv")
print(f"Results saved to {SAVE}")
