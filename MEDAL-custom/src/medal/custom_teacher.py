"""
Added to define custom teacher grammar with primary hyperparameter to be checked during MEDAL sweep.
Author: Claire He 
"""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np

BUILTIN_TEACHERS = frozenset({
    "umap", "tsne", "pca", "spectral", "phate", "isomap",
})

# Reserved for MEDAL's internal custom-teacher identifiers.
CUSTOM_PREFIX = "__custom__"


def json_value(value):
    """Convert supported metadata to deterministic JSON-compatible values."""
    if isinstance(value, np.generic):
        return json_value(value.item())

    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("Parameter dictionary keys must be strings.")
        return {k: json_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Parameter values must be finite.")
        return value

    raise TypeError(
        f"Unsupported parameter value {type(value).__name__}. "
        "Use JSON-compatible parameters."
    )


def stable_digest(value):
    payload = json.dumps(
        json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_custom_key(teacher):
    return isinstance(teacher, str) and teacher.startswith(CUSTOM_PREFIX)


@dataclass(frozen=True)
class SweepSpec:
    param_name: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    scale: str = "log"
    n_values: int = 15
    dtype: str = "float"
    values: Optional[Tuple[Any, ...]] = None

    def grid_values(self):
        if not isinstance(self.param_name, str) or not self.param_name:
            raise ValueError("param_name must be a nonempty string.")
        if self.param_name == "n_components":
            raise ValueError(
                "Custom sweeps currently require a fixed embedding dimension."
            )

        if self.values is not None:
            if self.minimum is not None or self.maximum is not None:
                raise ValueError(
                    "Provide explicit values OR range bounds, not both."
                )
            values = json_value(list(self.values))
        else:
            if self.minimum is None or self.maximum is None:
                raise ValueError("Both sweep bounds are required.")

            bounds = np.asarray(
                [self.minimum, self.maximum], dtype=float
            )
            if not np.isfinite(bounds).all():
                raise ValueError("Sweep bounds must be finite.")
            if self.minimum >= self.maximum:
                raise ValueError("minimum must be smaller than maximum.")
            if (
                isinstance(self.n_values, bool)
                or not isinstance(self.n_values, int)
                or self.n_values < 2
            ):
                raise ValueError("n_values must be an integer >= 2.")

            if self.scale == "log":
                if self.minimum <= 0:
                    raise ValueError("Logarithmic bounds must be positive.")
                raw = np.geomspace(
                    self.minimum, self.maximum, self.n_values
                )
            elif self.scale == "linear":
                raw = np.linspace(
                    self.minimum, self.maximum, self.n_values
                )
            else:
                raise ValueError("scale must be 'log' or 'linear'.")

            if self.dtype == "int":
                raw = np.rint(raw).astype(int)
            elif self.dtype != "float":
                raise ValueError("dtype must be 'int' or 'float'.")

            values = np.unique(raw).tolist()

        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(
                    "Sweep values must be finite numeric scalars."
                )

        if len(values) < 2:
            raise ValueError("A sweep requires at least two distinct values.")
        if len(set(values)) != len(values):
            raise ValueError("Sweep values must be unique.")

        return values

    def metadata(self):
        return {
            "param_name": self.param_name,
            "values": self.grid_values(),
        }


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    version: str
    embedding_fn: Callable
    sweep: SweepSpec
    default_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Teacher name must be a nonempty string.")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Teacher version must be a nonempty string.")
        if not callable(self.embedding_fn):
            raise TypeError("embedding_fn must be callable.")

        defaults = json_value(dict(self.default_params))

        if "n_components" in defaults:
            raise ValueError(
                "Specify embedding dimensionality through n_components."
            )
        if self.sweep.param_name not in defaults:
            raise ValueError(
                "default_params must include a representative value of "
                "the swept parameter for architecture search."
            )

        # Copy rather than retaining the caller's dictionary.
        object.__setattr__(self, "default_params", defaults)
        self.sweep.grid_values()

    @property
    def key(self):
        # Internal identity used by caching and checkpoint naming.
        identity = {"name": self.name, "version": self.version}
        return CUSTOM_PREFIX + stable_digest(identity)[:20]

    def resolve_params(self, overrides=None, n_components=2):
        params = json_value({
            **self.default_params,
            **dict(overrides or {}),
        })

        resolved_dim = params.pop("n_components", n_components)
        if resolved_dim != n_components:
            raise ValueError(
                "Teacher n_components must equal the student latent_dim."
            )

        return {**params, "n_components": n_components}

    def build_param_grid(self, n_components=2):
        return [
            self.resolve_params(
                {self.sweep.param_name: value},
                n_components=n_components,
            )
            for value in self.sweep.grid_values()
        ]

    def metadata(self):
        return {
            "name": self.name,
            "version": self.version,
            "key": self.key,
            "default_params": json_value(dict(self.default_params)),
            "sweep": self.sweep.metadata(),
        }


def resolve_custom_grid(spec, param_grid, n_components):
    """Resolve defaults and enforce one varying numeric parameter."""
    if param_grid is None:
        grid = spec.build_param_grid(n_components)
    else:
        grid = [
            spec.resolve_params(tc, n_components)
            for tc in param_grid
        ]

    if len(grid) < 2:
        raise ValueError("A custom sweep requires at least two configurations.")

    name = spec.sweep.param_name
    values = [tc[name] for tc in grid]

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError("The swept parameter must be finite and numeric.")

    if len(set(values)) != len(values):
        raise ValueError("Use one configuration per swept parameter value.")

    fixed = [
        {k: v for k, v in tc.items() if k != name}
        for tc in grid
    ]
    if any(tc != fixed[0] for tc in fixed[1:]):
        raise ValueError(
            "Only the primary parameter may vary within a custom sweep."
        )

    return grid


def compute_embedding(teacher, X, n_components=2, **params):
    """Dispatch without changing the existing built-in implementation."""
    if not isinstance(teacher, TeacherSpec):
        if is_custom_key(teacher):
            raise ValueError(
                "An internal custom teacher key cannot compute embeddings; "
                "provide the TeacherSpec."
            )

        from medal.teacher import get_teacher_embeddings
        return get_teacher_embeddings(
            teacher, X, n_components=n_components, **params
        )

    resolved = teacher.resolve_params(params, n_components)
    raw = np.asarray(teacher.embedding_fn(X, **resolved))

    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("Teacher coordinates must be real numeric values.")

    Z = raw.astype(np.float32)
    expected = (len(X), n_components)

    if Z.shape != expected:
        raise ValueError(
            f"Teacher returned shape {Z.shape}; expected {expected}."
        )
    if not np.isfinite(Z).all():
        raise ValueError("Teacher coordinates contain NaN or infinity.")

    # Reject a completely collapsed embedding.
    radius = np.sqrt(
        np.mean(np.sum((Z.astype(np.float64) - Z.mean(0)) ** 2, axis=1))
    )
    if not np.isfinite(radius) or radius <= 1e-8:
        raise ValueError("Teacher embedding has zero or negligible radius.")

    return Z