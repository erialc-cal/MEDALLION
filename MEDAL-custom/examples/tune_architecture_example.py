"""
Architecture search example for MEDAL.

Demonstrates the full workflow:
  1. Load data
  2. Search for the best AE architecture (tune_medal_architecture)
  3. Run a teacher hyperparameter sweep (run_teacher_sweep)
  4. Select the optimal teacher parameter (select_teacher_param)
  5. Visualise tuning curve (plot_reconstruction_error)
  6. Visualise distortion map at the optimal hyperparameter (plot_distortion_map)

Run with:
    python examples/tune_architecture_example.py
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from ray import tune

import medal

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading MNIST (first 5000 samples)…")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data[:5000].astype(np.float32) / 255.0
y = mnist.target[:5000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ── 2. Architecture search ────────────────────────────────────────────────────
# Default: grid search over hidden_dims × lr × lambda_d with a UMAP teacher.
# For a quick smoke-test, override the search space to a small subset:
small_space = {
    "hidden_dims": tune.grid_search([[512, 512], [512, 512, 512]]),
    "lr":          tune.grid_search([1e-3, 1e-4]),
    "lambda_d":    tune.grid_search([100, 1000]),
}

result = medal.tune_medal_architecture(
    X_train,
    teacher="umap",
    teacher_params={"n_neighbors": 15, "min_dist": 0.1},
    output_dir="output/arch_search",
    latent_dim=2,
    search_space=small_space,     # override: 2×2×2 = 8 trials
    search_mode="grid",
    num_samples=1,
    resources_per_trial={"cpu": 4, "gpu": 1},
    max_epochs=3000,               # short for the example
    save_results=True,
    verbose=True,
)

print("\n=== Best architecture ===")
print(result)
print("\nbest_config:", result.best_config)
print("best_metrics:", result.best_metrics)
print("Teacher embedding:", result.teacher_emb_path)
print("Results CSV saved to:", result.output_dir / "arch_search_results.csv")

# ── 3. Teacher hyperparameter sweep ──────────────────────────────────────────
arch_config = result.to_arch_config()

sweep_results = medal.run_teacher_sweep(
    X_train,
    output_dir="output/teacher_sweep",
    teacher="umap",
    arch_config=arch_config,
    latent_dim=2,
    seeds=[0, 1],
    verbose=True,
)

# ── 4. Select optimal teacher hyperparameter ─────────────────────────────────
df = sweep_results.load_metrics(X_test)
opt_n_neighbors = medal.select_teacher_param(df, param_col="n_neighbors")
print(f"\nOptimal n_neighbors: {opt_n_neighbors}")

# ── 5. Tuning curve ───────────────────────────────────────────────────────────
medal.plot_reconstruction_error(df, opt_n_neighbors, param_col="n_neighbors")

# ── 6. Distortion map at the optimal hyperparameter ──────────────────────────
# Load embeddings only for the selected parameter to keep this fast.
# y_train are labels for the full X_train passed to run_teacher_sweep;
# load_embeddings re-applies the same internal val split to match them.
emb_data = sweep_results.load_embeddings(
    y_train,
    X_test,
    y_test,
    params=[opt_n_neighbors],
)
medal.plot_distortion_map(
    emb_data,
    opt_n_neighbors,
    param_col="n_neighbors",
    param="best",
)
