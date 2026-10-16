import numpy as np
from typing import Tuple, Dict, Optional, Union
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def _ensure_images_shape(X: np.ndarray) -> Tuple[np.ndarray, bool]:
    X = np.asarray(X)
    if X.ndim == 2 and X.shape[1] == 784:
        X_img = X.reshape(-1, 28, 28)
        was_flattened = True
    elif X.ndim == 3 and X.shape[1:] == (28, 28):
        X_img = X
        was_flattened = False
    else:
        raise ValueError("X must be shape (N,784) or (N,28,28).")
    X_img = X_img.astype(np.float32)
    if X_img.max() > 1.0:
        X_img = X_img / 255.0
    X_img = np.clip(X_img, 0.0, 1.0)
    return X_img, was_flattened

def _restore_shape(X_img: np.ndarray, was_flattened: bool) -> np.ndarray:
    return X_img.reshape(len(X_img), -1) if was_flattened else X_img

# -------------------------
# Mask helpers
# -------------------------
def _normalize_mask(
    pixel_mask: Optional[np.ndarray],
    mask_frac: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Returns a boolean mask of shape (28,28), or None meaning "all pixels".
    If pixel_mask is provided, it is converted to boolean and reshaped if needed.
    If not provided, a random mask is generated using mask_frac.
    """
    if pixel_mask is None:
        if mask_frac >= 1.0:
            return None  # treat as "all pixels"
        if not (0.0 < mask_frac < 1.0):
            raise ValueError("mask_frac must be in (0,1] when pixel_mask is None.")
        m = rng.random((28, 28)) < mask_frac
        return m.astype(bool)

    m = np.asarray(pixel_mask)
    if m.shape == (784,):
        m = m.reshape(28, 28)
    if m.shape != (28, 28):
        raise ValueError("pixel_mask must have shape (28,28) or (784,).")
    return m.astype(bool)

def _maybe_per_image_mask(
    base_mask: Optional[np.ndarray],
    mask_mode: str,
    mask_frac: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    If mask_mode == 'per_image', ignore base_mask and sample a new mask.
    If 'shared', return base_mask as-is (could be None meaning all pixels).
    """
    if mask_mode not in ("shared", "per_image"):
        raise ValueError("mask_mode must be 'shared' or 'per_image'.")
    if mask_mode == "shared":
        return base_mask
    # per_image
    if mask_frac >= 1.0:
        return None
    m = rng.random((28, 28)) < mask_frac
    return m.astype(bool)

# -------------------------
# Masked transforms
# -------------------------
def _apply_noise(img: np.ndarray, sigma: float, rng: np.random.Generator, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
        return np.clip(img + noise, 0.0, 1.0)
    out = img.copy()
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    out[mask] = out[mask] + noise[mask]
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _apply_brightness_contrast(img: np.ndarray, alpha: float, beta: float, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return np.clip(alpha * img + beta, 0.0, 1.0).astype(np.float32)
    out = img.copy()
    out[mask] = alpha * out[mask] + beta
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _apply_smoothing(img: np.ndarray, n_iter: int = 1, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute smoothing update as before, but only write back into masked pixels.
    Note: the blur uses full neighborhood context (including unmasked pixels),
    but only masked pixels are changed.
    """
    if mask is None:
        out = img.copy()
        for _ in range(n_iter):
            up    = np.roll(out, shift=-1, axis=0)
            down  = np.roll(out, shift=+1, axis=0)
            left  = np.roll(out, shift=-1, axis=1)
            right = np.roll(out, shift=+1, axis=1)
            out = (out + up + down + left + right) / 5.0
        return out.astype(np.float32)

    out = img.copy()
    tmp = out.copy()
    for _ in range(n_iter):
        up    = np.roll(tmp, shift=-1, axis=0)
        down  = np.roll(tmp, shift=+1, axis=0)
        left  = np.roll(tmp, shift=-1, axis=1)
        right = np.roll(tmp, shift=+1, axis=1)
        blurred = (tmp + up + down + left + right) / 5.0
        # only update masked pixels
        tmp = tmp.copy()
        tmp[mask] = blurred[mask]
    out = tmp
    return out.astype(np.float32)

def _make_checkerboard(h: int = 28, w: int = 28, amp: float = 0.10) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    pat = ((xx + yy) % 2) * 2 - 1
    return (amp * pat).astype(np.float32)

def _apply_checker_watermark(
    img: np.ndarray,
    pattern: np.ndarray,
    clip: bool = True,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if mask is None:
        out = img + pattern
    else:
        out = img.copy()
        out[mask] = out[mask] + pattern[mask]
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)

# -------------------------
# Main function
# -------------------------
def make_artificial_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_fracs: Dict[str, float] = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
    noise_sigma: float = 0.20,
    alpha: float = 1.20,
    beta: float = 0.05,
    smooth_iters: int = 50,
    checker_amp: float = 0.10,
    checker_batch: str = "D",
    checker_clip: bool = True,
    seed: int = 0,
    stratify_by_label: bool = True,
    # --- NEW: pixel subset controls ---
    pixel_mask: Optional[np.ndarray] = None,   # (28,28) or (784,); True=can change
    mask_frac: float = 1.0,                    # used only if pixel_mask is None OR mask_mode="per_image"
    mask_mode: str = "shared",                 # "shared" or "per_image"
    mask_seed: int = 0,                        # separate seed for mask generation
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(sum(batch_fracs.values()) - 1.0) < 1e-6, "batch_fracs must sum to 1.0"
    X_img, was_flattened = _ensure_images_shape(X)
    y = np.asarray(y)
    assert len(X_img) == len(y), "X and y must have same length"

    rng = np.random.default_rng(seed)
    rng_mask = np.random.default_rng(mask_seed)

    N = len(X_img)
    batch_ids = np.empty(N, dtype=object)

    def assign_indices(idxs: np.ndarray, fracs: Dict[str, float]) -> Dict[str, np.ndarray]:
        n = len(idxs)
        counts = {k: int(round(fracs[k] * n)) for k in fracs}
        diff = n - sum(counts.values())
        if diff != 0:
            k_adj = max(fracs, key=fracs.get)
            counts[k_adj] += diff
        rng.shuffle(idxs)
        splits = {}
        start = 0
        for k in fracs:
            end = start + counts[k]
            splits[k] = idxs[start:end]
            start = end
        return splits

    if stratify_by_label:
        for digit in np.unique(y):
            idxs = np.where(y == digit)[0]
            splits = assign_indices(idxs, batch_fracs)
            for k, ss in splits.items():
                batch_ids[ss] = k
    else:
        idxs = np.arange(N)
        splits = assign_indices(idxs, batch_fracs)
        for k, ss in splits.items():
            batch_ids[ss] = k

    if checker_batch not in batch_fracs:
        raise ValueError(f"checker_batch={checker_batch!r} must be one of {list(batch_fracs.keys())}")
    checker_pattern = _make_checkerboard(28, 28, amp=checker_amp)

    # base mask (used if mask_mode="shared")
    base_mask = _normalize_mask(pixel_mask, mask_frac=mask_frac, rng=rng_mask)

    X_out = np.empty_like(X_img, dtype=np.float32)
    for b in ["A", "B", "C", "D"]:
        idxs = np.where(batch_ids == b)[0]
        if len(idxs) == 0:
            continue

        for ii in idxs:
            im = X_img[ii]
            mask = _maybe_per_image_mask(base_mask, mask_mode=mask_mode, mask_frac=mask_frac, rng=rng_mask)

            if b == checker_batch:
                X_out[ii] = _apply_checker_watermark(im, checker_pattern, clip=checker_clip, mask=mask)
            elif b == "A":
                X_out[ii] = im
            elif b == "B":
                X_out[ii] = _apply_noise(im, noise_sigma, rng, mask=mask)
            elif b == "C":
                X_out[ii] = _apply_brightness_contrast(im, alpha, beta, mask=mask)
            elif b == "D":
                X_out[ii] = _apply_smoothing(im, n_iter=smooth_iters, mask=mask)
            else:
                raise ValueError(f"Unknown batch key {b}")

    X_aug = _restore_shape(X_out, was_flattened)
    y_bio = y.copy()
    y_batch = batch_ids.astype(str)
    return X_aug, y_bio, y_batch