import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dims=(128, 128),
                 activation=nn.ReLU, bottleneck_activation=None, dropout_rate=0.1,
                 use_batchnorm=False, final_activation=None):
        super().__init__()
        self.ActivationCls = activation
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm

        # --- ENCODER ---
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            if activation is not None:
                encoder_layers.append(activation())
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        if bottleneck_activation is not None:
            encoder_layers.append(bottleneck_activation())
        self.encoder = nn.Sequential(*encoder_layers)

        # --- DECODER ---
        decoder_layers = []
        prev_dim = latent_dim
        decoder_layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm1d(hidden_dims[-1]))
        if activation is not None:
            decoder_layers.append(activation())
        if self.dropout_rate > 0:
            decoder_layers.append(nn.Dropout(self.dropout_rate))
        prev_dim = hidden_dims[-1]
        for h in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                decoder_layers.append(nn.BatchNorm1d(h))
            if activation is not None:
                decoder_layers.append(activation())
            if self.dropout_rate > 0:
                decoder_layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        if final_activation is not None:
            decoder_layers.append(final_activation())
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _init_weights(self, eps=1e-3):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.use_batchnorm:
                    nn.init.normal_(m.weight, mean=0.0, std=eps)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class MEDAL(BaseEstimator, TransformerMixin):
    """
    Manifold Embedding Distillation via Autoencoder Learning.

    Trains an autoencoder to simultaneously reconstruct X and match a
    pre-computed teacher embedding Z via a distillation loss.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input data.
    latent_dim : int
        Dimensionality of the bottleneck (target embedding space).
    hidden_dims : tuple of int
        Width of each hidden layer (symmetric encoder/decoder).
    activation : str or nn.Module class
        Hidden-layer activation.  Pass a string (e.g. "ReLU", "SELU") or
        a torch.nn class directly.
    lambda_d : float
        Weight on the distillation loss relative to reconstruction loss.
    lr : float
        Initial learning rate for AdamW.
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size.
    """

    def __init__(self, input_dim, latent_dim=2, hidden_dims=(128, 64),
                 activation="ReLU", bottleneck_activation=None,
                 final_activation=None, criterion=nn.MSELoss,
                 lambda_d=10, lr=1e-3, epochs=100, batch_size=32,
                 eta_min=1e-16, device=None, clip_grad_norm=1.0,
                 warmup=0, adamw_weight_decay=1e-5, patience=20,
                 factor=0.9, use_batchnorm=False, dropout_rate=0.1,
                 **kwargs):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = getattr(nn, activation) if isinstance(activation, str) else activation
        self.bottleneck_activation = (
            getattr(nn, bottleneck_activation)
            if isinstance(bottleneck_activation, str) else bottleneck_activation
        )
        self.final_activation = (
            getattr(nn, final_activation)
            if isinstance(final_activation, str) else final_activation
        )
        self.lambda_d = lambda_d
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_grad_norm = clip_grad_norm
        self.warmup = warmup
        self.eta_min = eta_min
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

        self.model = AutoEncoder(
            input_dim, latent_dim, hidden_dims,
            activation=self.activation,
            bottleneck_activation=self.bottleneck_activation,
            final_activation=self.final_activation,
            dropout_rate=self.dropout_rate,
            use_batchnorm=self.use_batchnorm,
        ).to(self.device)

        self.opt_joint = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=adamw_weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_joint, "min", factor=factor, threshold=1e-4,
            patience=patience, min_lr=self.eta_min, eps=1e-15,
        )
        self.criterion = criterion().to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, teacher_Z=None, verbose=True,
            target_bands=None, stability_window=10,
            epsilon_distill=0.1, epsilon_recon=0.005,
            patience=3, return_on_stable=False,
            save_dir=None, prefix=None,
            print_tag=False):
        """
        Train the MEDAL autoencoder.

        Parameters
        ----------
        X : array-like of shape (n_samples, input_dim)
            Input data.
        teacher_Z : array-like of shape (n_samples, latent_dim), optional
            Pre-computed (and normalised) teacher embeddings.  When omitted,
            the model trains as a plain autoencoder (no distillation loss).
        verbose : bool
            Show tqdm progress bar.
        target_bands : list of (float, float), optional
            Distillation-loss bands for stability-based early stopping.
        stability_window : int
            Epochs over which to measure loss slope for stability check.
        epsilon_distill, epsilon_recon : float
            Maximum allowed loss slope for the stability condition.
        patience : int
            Consecutive stable checks required before early stopping.
        return_on_stable : bool
            Stop as soon as stability is confirmed in the final band.
        save_dir : str or Path, optional
            Directory to write checkpoints into.
        prefix : str, optional
            Filename prefix for checkpoint files.
        print_tag : bool
            Print metrics to stdout instead of reporting to Ray Tune.
        """

        def _report_or_print(metrics):
            if print_tag:
                print(metrics)
            else:
                try:
                    from ray import tune as _tune
                    _tune.report(metrics)
                except RuntimeError:
                    pass  # not inside a Ray Tune trial

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if teacher_Z is not None:
            teacher_Z = torch.tensor(teacher_Z, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X, teacher_Z) if teacher_Z is not None else TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        report_interval = max(1, self.epochs // 1000)

        distill_history, recon_history = [], []
        self.stable_counter = 0
        early_stopped = False
        target_bands_saved = np.zeros(len(target_bands)) if target_bands else None
        self.stable_band_ = None
        self.stable_epoch_ = None
        best_recon_in_band = {}

        for epoch in tqdm(range(self.epochs), disable=not verbose):
            epoch_distill_loss = 0.0
            epoch_recon_loss = 0.0
            num_batches = 0

            for batch in loader:
                x = batch[0].to(self.device, non_blocking=True)
                teacher_z = (
                    batch[1].to(self.device, non_blocking=True)
                    if teacher_Z is not None else None
                )
                self.opt_joint.zero_grad()
                x_rec, z = self.model(x)
                recon_loss = self.criterion(x_rec, x)

                if teacher_z is None:
                    loss = recon_loss
                    distill_loss = torch.tensor(0.0, device=self.device)
                else:
                    lam = (
                        self.lambda_d * (epoch / self.warmup)
                        if epoch < self.warmup else self.lambda_d
                    )
                    distill_loss = self.criterion(z, teacher_z)
                    loss = recon_loss + lam * distill_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.opt_joint.step()

                epoch_distill_loss += distill_loss.item()
                epoch_recon_loss += recon_loss.item()
                num_batches += 1

            avg_distill = epoch_distill_loss / num_batches
            avg_recon = epoch_recon_loss / num_batches
            distill_history.append(avg_distill)
            recon_history.append(avg_recon)
            self.scheduler.step(avg_distill if teacher_Z is not None else avg_recon)

            if target_bands:
                early_stopped = self._check_stability_and_checkpoint(
                    epoch, avg_distill, avg_recon, distill_history, recon_history,
                    target_bands, stability_window, epsilon_distill, epsilon_recon,
                    patience, return_on_stable, save_dir, prefix,
                    target_bands_saved, best_recon_in_band,
                )

            if (epoch + 1) % report_interval == 0 or epoch == self.epochs - 1:
                _report_or_print({
                    'distill_loss': avg_distill,
                    'recon_loss': avg_recon,
                    'lr': self.opt_joint.param_groups[0]['lr'],
                    'stability': self.stable_counter,
                })

            if early_stopped:
                break

        # Store training stats for retrieval by callers (e.g. sweep summary)
        self.n_epochs_trained_ = epoch + 1
        self.final_distill_loss_ = distill_history[-1] if distill_history else float("nan")
        self.final_recon_loss_ = recon_history[-1] if recon_history else float("nan")

        if save_dir is not None:
            base = Path(save_dir) / f"{prefix}_ckpts"
            base.mkdir(parents=True, exist_ok=True)
            ckpt_path = base / "final.pt"
            torch.save({"model": self._state_dict_cpu()}, ckpt_path)
            print(f"Saved model to {ckpt_path}")

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(self, X):
        """Return latent embeddings Z for input X.

        Parameters
        ----------
        X : array-like of shape (n_samples, input_dim)

        Returns
        -------
        Z : np.ndarray of shape (n_samples, latent_dim)
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X, dtype=np.float32)).to(self.device)
            _, Z = self.model(X_t)
        return Z.cpu().numpy()

    def reconstruct(self, X):
        """Return reconstructed X for input X.

        Parameters
        ----------
        X : array-like of shape (n_samples, input_dim)

        Returns
        -------
        X_recon : np.ndarray of shape (n_samples, input_dim)
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X, dtype=np.float32)).to(self.device)
            X_recon, _ = self.model(X_t)
        return X_recon.cpu().numpy()

    # ------------------------------------------------------------------
    # sklearn TransformerMixin compatibility
    # ------------------------------------------------------------------

    def fit_transform(self, X, teacher_Z=None, **fit_kwargs):
        self.fit(X, teacher_Z, **fit_kwargs)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state_dict_cpu(self):
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def _check_stability_and_checkpoint(
        self, epoch, avg_distill, avg_recon,
        distill_history, recon_history,
        target_bands, stability_window,
        epsilon_distill, epsilon_recon, patience,
        return_on_stable, save_dir, prefix,
        target_bands_saved, best_recon_in_band,
    ):
        current_band = None

        for idx, (tau_min, tau_max) in enumerate(target_bands):
            if tau_min <= avg_distill <= tau_max:
                current_band = idx
                if not target_bands_saved[idx] and idx < len(target_bands) - 1:
                    base = Path(save_dir) / f"{prefix}_ckpts"
                    base.mkdir(parents=True, exist_ok=True)
                    ckpt_path = base / f"band{idx}.pt"
                    torch.save({"model": self._state_dict_cpu()}, ckpt_path)
                    print(f"Saved band checkpoint to {ckpt_path} (distill={avg_distill:.6f}, epoch={epoch})")
                    target_bands_saved[idx] = 1
                break

        early_stop_flag = False

        if current_band is not None and len(distill_history) >= stability_window:
            slope_dist = abs(distill_history[-1] - distill_history[-stability_window]) / stability_window
            slope_recon = abs(recon_history[-1] - recon_history[-stability_window]) / stability_window
            is_stable = (slope_dist < epsilon_distill) and (slope_recon < epsilon_recon)

            if is_stable:
                self.stable_counter += 1
            else:
                self.stable_counter = 0

            if self.stable_counter >= patience:
                self.stable_band_ = current_band
                self.stable_epoch_ = epoch
                prev = best_recon_in_band.get(current_band)
                if prev is None or avg_recon < prev[0] - 1e-12:
                    best_recon_in_band[current_band] = (avg_recon, None)
                    if return_on_stable and current_band == len(target_bands) - 1:
                        early_stop_flag = True
        else:
            self.stable_counter = 0

        return early_stop_flag
