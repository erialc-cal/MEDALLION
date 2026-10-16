import numpy as np
import pickle
from pathlib import Path


class GlobalEmbeddingNormalizer:
    """
    Normalises embeddings to zero mean and unit RMS radius.
    Fit on training data; apply the same transform to val/test.
    """
    def __init__(self, mean_=None, scale_=None, eps=1e-8):
        self.mean_ = mean_
        self.scale_ = scale_
        self.eps = eps

    def fit(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        self.mean_ = Z.mean(axis=0, keepdims=True)
        Zc = Z - self.mean_
        self.scale_ = np.sqrt(np.mean(np.sum(Zc**2, axis=1)))
        self.scale_ = max(float(self.scale_), self.eps)
        return self

    def transform(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        return (Z - self.mean_) / self.scale_

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        return Z * self.scale_ + self.mean_

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean_, "scale": self.scale_}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(
            mean_=np.asarray(d["mean"], dtype=np.float32),
            scale_=float(d["scale"]),
        )
