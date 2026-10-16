"""
Added to keep track of custom teacher embeddings. 
Validation of custom experiment output-directory reuse.
"""
from pathlib import Path
import json
import os
import tempfile

import numpy as np

from medal.custom_teacher import json_value, stable_digest


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(json_value(payload), handle, indent=2)

    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def data_digest(X):
    import hashlib

    X = np.asarray(X)
    if X.ndim != 2 or not np.issubdtype(X.dtype, np.number):
        raise ValueError("Expected a two-dimensional numeric input array.")
    if np.iscomplexobj(X):
        raise ValueError("Complex inputs are unsupported.")

    digest = hashlib.sha256()
    digest.update(
        f"{X.shape}:{X.dtype.str}".encode("utf-8")
    )

    for start in range(0, len(X), 1024):
        chunk = np.ascontiguousarray(X[start:start + 1024])
        if not np.isfinite(chunk).all():
            raise ValueError("Input data contains NaN or infinity.")
        digest.update(chunk.tobytes())

    return digest.hexdigest()


def guard_custom_output(output_dir, X, settings):
    """Reject incompatible reuse before overwriting experiment files."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "custom_run_identity.json"

    identity = json_value({
        "metadata_version": 1,
        "data_digest": data_digest(X),
        "settings": settings,
    })

    if path.exists():
        with path.open() as handle:
            previous = json.load(handle)

        if stable_digest(previous) != stable_digest(identity):
            raise ValueError(
                "This output directory belongs to different inputs or "
                "settings. Use a new output_dir."
            )
    else:
        if any(root.iterdir()):
            raise ValueError(
                "Custom output directory is nonempty but lacks provenance. "
                "Use a new output_dir."
            )
        atomic_json(path, identity)