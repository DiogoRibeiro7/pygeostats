# src/python/pygeostats/kriging/_solver.py
"""Kriging systems that are factorised once and reused by every prediction.

``predict`` used to build and LU-factorise the whole kriging system on every call,
then solve it once per target, in a single thread and holding the GIL. With many
samples the factorisation dominated, and splitting the targets into batches or
executor chunks repeated it for each one.

The system is LU-factorised when an estimator is fitted instead, in the Rust core.
Because the system is symmetric, one solve against the sample values gives dual
weights, and the prediction at any target is then a covariance-weighted sum over
the samples, which the Rust core evaluates in parallel with the GIL released. The
kriging variance still needs a solve for each target, which the Rust core does in
parallel from the stored factors, for each target independently of the others.
"""

# The factorisation is in the Rust core rather than SciPy or NumPy. On a 22-core
# Windows machine, SciPy 1.15.3 with its bundled OpenBLAS 0.3.28 took 0.55 to 1.6 s
# to factorise a 501-square system, and its lu_solve returned wrong solutions when
# called from several threads at once: 739 of 800 calls from 8 threads. Inverting
# the system with NumPy was fast, but variances computed from the inverse moved by
# up to 4e-5 depending on how many targets were computed together.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .._core import (
    covariance_between,
    dual_kriging_predict,
    factorised_kriging_variance,
    lu_factorize,
    lu_solve,
)


def variogram_parameters(variogram) -> np.ndarray:
    """The ``[nugget, sill, range]`` of a fitted isotropic variogram."""
    return np.array([variogram.nugget_, variogram.sill_, variogram.range_], dtype=float)


@dataclass(frozen=True)
class KrigingSystem:
    """An LU-factorised kriging system for one set of samples and parameters.

    The first ``n_samples`` rows and columns hold the covariances between samples;
    any rows and columns after them hold drift constraints, such as the
    unbiasedness condition of ordinary kriging.
    """

    coordinates: np.ndarray
    parameters: np.ndarray
    model: str
    lu: np.ndarray
    permutation: np.ndarray

    @classmethod
    def build(
        cls,
        coordinates: np.ndarray,
        parameters: np.ndarray,
        model: str,
        drift: Optional[np.ndarray] = None,
    ) -> KrigingSystem:
        """Build and factorise the system, raising ``ValueError`` if it is singular."""
        coordinates = np.ascontiguousarray(coordinates, dtype=float)
        parameters = np.asarray(parameters, dtype=float)
        covariance = covariance_between(coordinates, coordinates, parameters, model)
        if drift is None:
            matrix = covariance
        else:
            n_samples, n_drift = drift.shape
            matrix = np.zeros((n_samples + n_drift, n_samples + n_drift))
            matrix[:n_samples, :n_samples] = covariance
            matrix[:n_samples, n_samples:] = drift
            matrix[n_samples:, :n_samples] = drift.T
        lu, permutation = lu_factorize(matrix)
        return cls(coordinates, parameters, model, lu, permutation)

    @property
    def n_samples(self) -> int:
        return len(self.coordinates)

    def matches(self, parameters: np.ndarray, model: str) -> bool:
        """Whether the system was built for these variogram parameters."""
        return self.model == model and np.array_equal(
            self.parameters, np.asarray(parameters, dtype=float)
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve the system for one right-hand side."""
        return lu_solve(
            self.lu, self.permutation, np.ascontiguousarray(rhs, dtype=float)
        )

    def covariance_sum(self, targets: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """For each target, the sum over samples of covariance times ``weights``."""
        return dual_kriging_predict(
            self.coordinates,
            np.ascontiguousarray(targets, dtype=float),
            self.parameters,
            self.model,
            np.ascontiguousarray(weights, dtype=float),
        )


def ordinary_system(
    coordinates: np.ndarray, parameters: np.ndarray, model: str
) -> KrigingSystem:
    """The ordinary kriging system: covariances bordered by a row and column of ones."""
    return KrigingSystem.build(
        coordinates, parameters, model, drift=np.ones((len(coordinates), 1))
    )


def ordinary_variance(system: KrigingSystem, targets: np.ndarray) -> np.ndarray:
    """Ordinary kriging variance at each target, from a factorised ordinary system.

    For a target with covariances ``c`` to the samples, the system gives weights
    ``w`` and a Lagrange multiplier ``mu``, and the variance is
    ``sill - w @ c - mu``, floored at zero.
    """
    return factorised_kriging_variance(
        system.coordinates,
        np.ascontiguousarray(targets, dtype=float),
        system.parameters,
        system.model,
        system.lu,
        system.permutation,
    )
