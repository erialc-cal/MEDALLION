from pathlib import Path
from copy import deepcopy
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from medal.io import load_model

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(4)
from sklearn.datasets import load_digits

# Load data
folder = Path("/Users/clairehe/Documents/GitHub/MEDAL_FI/encoder_FI/experiments/arch_search")
X_train, X_val = np.load(folder / "X_train.npy").astype(np.float32), np.load(folder / 'X_val.npy').astype(np.float32)
X = torch.from_numpy(X_train)
V = torch.from_numpy(X_val)
teach_embed = np.load( folder / "embeddings/arch_search_teacher.npy").astype(np.float32)
Z = torch.from_numpy(teach_embed)
digits = load_digits()
label_lookup = {tuple(image): label for image, label in zip(digits.data, digits.target)}
labels = np.array([label_lookup[tuple(image)] for image in X_train])

# Load medal's encoder 
OUT = Path('../data/baby_mnist_medal')
cfg = json.loads((OUT / 'config.json').read_text())
model = medal.load_model(OUT / 'baby_mnist_ckpts/final.pt', input_dim=64, hidden_dims=tuple(cfg['hidden_dims']), latent_dim=2, activation=cfg['activation'], use_batchnorm=cfg['use_batchnorm'], dropout_rate=cfg['dropout_rate'])
model.eval()

# Copy architecture, then initialize fresh weights.
encoder = deepcopy(model.encoder)
for module in encoder.modules():
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()

loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X, Z),
    batch_size=cfg["batch_size"],
    shuffle=True)

optimizer = torch.optim.AdamW(
    encoder.parameters(),
    lr=cfg["lr"],
    weight_decay=cfg["adamw_weight_decay"])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=cfg["t_factor"],
    patience=cfg["t_patience"],
    min_lr=cfg["eta_min"])

# Train ONLY with distillation loss; there is no decoder.
best_mse = float("inf")
best_state = None

for epoch in range(cfg["max_epochs"]):
    encoder.train()
    for xb, zb in loader:
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(encoder(xb), zb)
        loss.backward()
        optimizer.step()

    encoder.eval()
    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(encoder(X), Z).item()

    scheduler.step(mse)
    if mse < best_mse:
        best_mse = mse
        best_state = deepcopy(encoder.state_dict())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: distillation MSE = {mse:.8g}")

encoder.load_state_dict(best_state)
encoder.eval()

with torch.no_grad():
    medal_mse = torch.nn.functional.mse_loss(model.encoder(X), Z).item()

print(f"MEDAL training distillation MSE:        {medal_mse:.8g}")
print(f"Encoder-only training distillation MSE: {best_mse:.8g}")


def attribute(model, inputs, n_steps=2000):
    """Signed IG for each latent coordinate: (images, 2, 784)."""
    ig = IntegratedGradients(model)
    coordinates, errors = [], []

    for coordinate in range(2):
        attrs, deltas = [], []
        for batch in inputs.split(32):
            a, d = ig.attribute(
                batch,
                baselines=torch.zeros_like(batch),
                target=coordinate,
                n_steps=n_steps,
                internal_batch_size=128,
                return_convergence_delta=True,
            )
            attrs.append(a.detach().numpy())
            deltas.append(d.detach().numpy())

        coordinates.append(np.concatenate(attrs))
        errors.append(np.concatenate(deltas))

    return np.stack(coordinates, axis=1), np.stack(errors, axis=1)


# Recompute both using identical IG settings.
medal_ig, medal_delta = attribute(model.encoder, V)
encoder_ig, encoder_delta = attribute(encoder, V)
encoder_ig_train, encoder_delta_train = attribute(encoder, X)

print("Mean absolute IG completeness error:")
print(f"  MEDAL:        {np.abs(medal_delta).mean():.8g}")
print(f"  Encoder-only: {np.abs(encoder_delta).mean():.8g}")

# Compare signed attributions, preserving the two teacher coordinates.
a = medal_ig.reshape(len(V), -1)
b = encoder_ig.reshape(len(V), -1)
denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
valid = denominator > 0
cosine = np.full(len(V), np.nan)
cosine[valid] = (a[valid] * b[valid]).sum(axis=1) / denominator[valid]
print(f"Mean signed IG cosine similarity: {np.nanmean(cosine):.4f}")

# Aggregate coordinate magnitudes for visual comparison.
medal_maps = np.abs(medal_ig).sum(axis=1)
encoder_maps = np.abs(encoder_ig).sum(axis=1)

root = Path('experiments/arch_search/comp_files')
torch.save(best_state, root / "encoder_only.pt")
np.savez_compressed(
    root / "medal_vs_encoder_ig.npz",
    labels=labels,
    medal_ig=medal_ig,
    encoder_ig=encoder_ig,
    encoder_train_ig=encoder_ig_train,
    medal_delta=medal_delta,
    encoder_delta=encoder_delta,
    signed_cosine=cosine,
)