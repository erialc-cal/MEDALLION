<div align="center">
  <img src="medal_logo_icon_only.svg" alt="MEDAL logo" width="400">
</div>

# MEDAL

[![DOI](https://zenodo.org/badge/1136663495.svg)](https://doi.org/10.5281/zenodo.20347572)

**Manifold Embedding Distillation via Autoencoder Learning**

MEDAL trains a neural network autoencoder to *distill* a pre-computed manifold embedding (e.g. t-SNE, UMAP) into a fast, reusable encoder. Given high-dimensional data `X` and a teacher embedding `Z`, the student minimises a combined objective:

```
Loss = Reconstruction Loss + λ_d × Distillation Loss
```

The result is an encoder that maps new data points directly to a low-dimensional space that mirrors the structure of the original teacher embedding — without re-running the teacher algorithm.

---

## Installation

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate medal

# Install the package in editable mode
pip install -e .
```

**For GPU support**, add `pytorch-cuda=<your-cuda-version>` to `environment.yml` before running the above.

**Dependencies:** Python 3.10, PyTorch, scikit-learn, Ray Tune, UMAP, openTSNE, PHATE, pandas, matplotlib.

---

## Quickstart

The standard MEDAL workflow has four steps:

```python
import medal

# 1. Find the best autoencoder architecture for your data
result = medal.tune_medal_architecture(
    X_train,
    teacher="tsne",
    teacher_params={"perplexity": 30},
    output_dir="experiments/arch_search",
)
arch_config = result.to_arch_config()

# 2. Sweep teacher hyperparameters, training one student per setting
sweep_results = medal.run_teacher_sweep(
    X_train,
    output_dir="experiments/sweep",
    teacher="tsne",
    arch_config=arch_config,
)

# 3. Evaluate and select the optimal hyperparameter
df = sweep_results.load_metrics(X_test)
opt_param = medal.select_teacher_param(df, param_col="perplexity")
print(f"Optimal perplexity: {opt_param}")

# 4. Visualise reconstruction error across the sweep
medal.plot_reconstruction_error(df, opt_param, param_col="perplexity")
```

A complete runnable example using MNIST is provided in [`examples/tune_architecture_example.py`](examples/tune_architecture_example.py).

---

## Workflow

### Step 1 — Architecture Search

`tune_medal_architecture` uses [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) to search over autoencoder architectures (hidden layer widths, learning rate, and `λ_d`). It computes the teacher embedding once and reuses it across all trials.

```python
result = medal.tune_medal_architecture(
    X_train,
    teacher="umap",                              # teacher algorithm
    teacher_params={"n_neighbors": 15, "min_dist": 0.1},
    output_dir="experiments/arch_search",
    latent_dim=2,                                # bottleneck dimensionality
    search_mode="grid",                          # "grid" or "random"
    num_samples=1,                               # trials per grid point (for random search)
    resources_per_trial={"cpu": 4, "gpu": 1},
    max_epochs=3000,
    save_results=True,
)

print(result.best_config)    # winning architecture dict
print(result.best_metrics)   # final distill_loss, recon_loss, lr
result.results_df            # pd.DataFrame — one row per trial
```

To override the default search space, pass a `search_space` dict using `ray.tune` samplers:

```python
from ray import tune

small_space = {
    "hidden_dims": tune.grid_search([[512, 512], [512, 512, 512]]),
    "lr":          tune.grid_search([1e-3, 1e-4]),
    "lambda_d":    tune.grid_search([100, 1000]),
}

result = medal.tune_medal_architecture(X_train, search_space=small_space, ...)
```

**`ArchSearchResults` attributes:**

| Attribute | Description |
|---|---|
| `best_config` | Winning architecture dict, ready to pass to `run_teacher_sweep` |
| `best_metrics` | Final metrics from the best trial |
| `results_df` | Per-trial results as a DataFrame |
| `teacher_emb_path` | Path to the cached teacher embedding (`.npy`) |
| `output_dir` | Root directory for all outputs |

Call `result.to_arch_config()` to get a clean config dict for Step 2, or `result.save()` / `ArchSearchResults.load(path)` to persist and reload results.

---

### Step 2 — Teacher Hyperparameter Sweep

`run_teacher_sweep` takes the best architecture from Step 1 and trains one student model per value of the teacher's main hyperparameter (e.g. `perplexity` for t-SNE, `n_neighbors` for UMAP), repeated across multiple random seeds for robustness.

```python
sweep_results = medal.run_teacher_sweep(
    X_train,
    output_dir="experiments/sweep",
    teacher="tsne",
    arch_config=arch_config,          # from result.to_arch_config()
    latent_dim=2,
    seeds=[0, 1, 2],                  # train each setting with 3 seeds
    normalize_teacher=True,           # normalise teacher embeddings
    verbose=True,
)
```

Teacher embeddings are computed and cached once per hyperparameter value; student checkpoints are saved for every `(param_value, seed)` pair.

**`SweepResults` attributes:**

| Attribute | Description |
|---|---|
| `output_dir` | Root directory for embeddings and checkpoints |
| `teacher` | Teacher algorithm name |
| `param_name` | Name of the swept hyperparameter |
| `param_values` | Sorted list of swept values |
| `seeds` | Random seeds used |
| `arch_config` | Architecture config shared by all students |

Call `sweep_results.save()` / `SweepResults.load(path)` to persist and reload.

---

### Step 3 — Evaluate and Select

`SweepResults.load_metrics` loads all student checkpoints and computes reconstruction and distillation MSE on one or more data splits.

```python
df = sweep_results.load_metrics(X_test)
# Or with explicit train/val/test splits:
df = sweep_results.load_metrics(X_train, X_val, X_test)
```

`select_teacher_param` then applies a two-step selection rule:

1. **Convergence filter** — discard `(param, seed)` pairs where training distillation MSE never fell below `distill_threshold`, indicating the model did not converge.
2. **One-SEM rule** — among converged models, pick the *smallest* hyperparameter value whose mean validation reconstruction loss is within one standard error of the global minimum.

```python
opt_param = medal.select_teacher_param(
    df,
    param_col="perplexity",      # column name of the swept hyperparameter
    metric_col="recon_loss",     # metric used for selection (default)
    val_split="Val",             # which split to optimise on
    distill_threshold=1e-5,      # convergence criterion
)
```

---

### Step 4 — Visualise

```python
medal.plot_reconstruction_error(df, opt_param, param_col="perplexity")
```

This plots reconstruction loss vs. hyperparameter value for train, validation, and test splits, with a marker at the selected optimum.

---

## Lower-Level API

For more control, you can use MEDAL components individually.

### Training a single model

```python
from medal import MEDAL

model = MEDAL(
    input_dim=784,
    latent_dim=2,
    hidden_dims=(512, 512, 512),
    lambda_d=1000,
    lr=1e-3,
    epochs=3000,
)

model.fit(X_train, teacher_Z=Z_train)   # Z_train: pre-computed teacher embedding
Z_pred  = model.transform(X_test)       # encode to latent space
X_recon = model.reconstruct(X_test)     # reconstruct original data
```

`MEDAL` is sklearn-compatible (`BaseEstimator`, `TransformerMixin`) and supports `fit_transform`.

### Computing teacher embeddings directly

```python
Z = medal.get_teacher_embeddings(
    method="umap",
    X=X_train,
    n_components=2,
    save_path="embeddings/umap_15.npy",   # optional caching
    n_neighbors=15,
    min_dist=0.1,
)
```

### Loading a saved model for inference

```python
model = medal.load_model(
    ckpt_path="experiments/sweep/student.pt",
    input_dim=784,
    hidden_dims=(512, 512, 512),
    latent_dim=2,
)

Z = medal.embed(model, X_new)             # returns numpy array
losses = medal.compute_losses(model, X, Z_teacher)
# losses: {"recon_loss": float, "distill_loss": float}
```

<!--  ### Normalising teacher embeddings

`GlobalEmbeddingNormalizer` centres embeddings to zero mean and scales to unit RMS radius, which can improve numerical stability during distillation.

```python
from medal import GlobalEmbeddingNormalizer

norm = GlobalEmbeddingNormalizer()
Z_norm = norm.fit(Z_train).transform(Z_train)
Z_test_norm = norm.transform(Z_test)

norm.save("experiments/normalizer.pkl")
norm = GlobalEmbeddingNormalizer.load("experiments/normalizer.pkl") 
``` -->

---

## Package Structure

```
src/medal/
├── model.py        # AutoEncoder (nn.Module) and MEDAL (sklearn estimator)
├── teacher.py      # get_teacher_embeddings, build_param_grid
├── tuning.py       # tune_medal_architecture, ArchSearchResults
├── sweep.py        # run_teacher_sweep, SweepResults
├── selection.py    # select_teacher_param, plot_reconstruction_error
├── normalizer.py   # GlobalEmbeddingNormalizer
├── io.py           # load_model, embed, compute_losses, eval_student
└── _paths.py       # Internal path helpers
```
