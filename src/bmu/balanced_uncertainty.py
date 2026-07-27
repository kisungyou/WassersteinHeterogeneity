#!/usr/bin/env python3
"""Balanced inference for pairwise uncertainty summaries.

This module implements the equireplicate pair design and cross-fitted
covariance estimator used in ``Quantifying between-measure uncertainty in
Wasserstein space``.  It accepts either a scalar kernel matrix or a vector of
pairwise component kernels.  The canonical Wasserstein total uses

    h_ij = 0.5 * W2(mu_i, mu_j)**2.

For an even sample size n, K edge-disjoint perfect matchings select n*K/2
pairs.  The raw first- and second-order covariance estimators are exactly
unbiased under the factor design before positive-semidefinite projection:

    E(Gamma1_raw) = Gamma1,
    E(Gamma2_raw) = Gamma2.

The selected-edge sample covariance requires the finite-design correction

    c_nK = 2 - 2*(K-1)/(M-1),  M=n*K/2,

rather than the complete-pair coefficient 2.

The public API randomizes vertex labels and the selected factor order; this is
the canonical released procedure.  The implementation assumes even n>4 and
even K for cross-fitted studentization.  A rotating-bye construction for odd
n is described in the manuscript but is not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log
from statistics import NormalDist
from typing import Any, Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class BalancedResult:
    """Output of :func:`balanced_inference_from_kernel`.

    All parameter-like quantities are one-dimensional arrays, even for a
    scalar kernel.  ``row_projection`` estimates the first Hoeffding
    projection g(X_i).  For h=W2^2/2, twice this array estimates the
    uncertainty-attribution score.
    """

    estimate: Array
    covariance: Array
    standard_error: Array
    confidence_interval: Array
    gamma1: Array
    gamma2: Array
    gamma1_raw: Array
    gamma2_raw: Array
    edge_covariance: Array
    edge_covariance_coefficient: float
    row_projection: Array
    half_projection_1: Array
    half_projection_2: Array
    selected_edges: Array
    edge_values: Array
    factor_order: Array
    relabeling: Array
    matchings: int
    evaluated_pairs: int
    confidence: float

    def scalar(self) -> dict[str, Any]:
        """Return a convenient dictionary when the kernel is scalar."""
        if self.estimate.size != 1:
            raise ValueError("scalar() is available only for a scalar kernel")
        return {
            "estimate": float(self.estimate[0]),
            "variance": float(self.covariance[0, 0]),
            "standard_error": float(self.standard_error[0]),
            "confidence_interval": tuple(
                float(v) for v in self.confidence_interval[0]
            ),
            "gamma1": float(self.gamma1[0, 0]),
            "gamma2": float(self.gamma2[0, 0]),
            "gamma1_raw": float(self.gamma1_raw[0, 0]),
            "gamma2_raw": float(self.gamma2_raw[0, 0]),
            "edge_variance": float(self.edge_covariance[0, 0]),
            "edge_covariance_coefficient": self.edge_covariance_coefficient,
            "row_projection": self.row_projection[:, 0].copy(),
            "half_projection_1": self.half_projection_1[:, 0].copy(),
            "half_projection_2": self.half_projection_2[:, 0].copy(),
            "selected_edges": self.selected_edges.copy(),
            "edge_values": self.edge_values[:, 0].copy(),
            "factor_order": self.factor_order.copy(),
            "relabeling": self.relabeling.copy(),
            "matchings": self.matchings,
            "evaluated_pairs": self.evaluated_pairs,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CompleteResult:
    """Regular complete-pair inference based on the first projection only."""

    estimate: Array
    covariance: Array
    standard_error: Array
    confidence_interval: Array
    gamma1: Array
    row_projection: Array
    evaluated_pairs: int
    confidence: float


def one_factor_matching(n: int, factor: int) -> tuple[Array, Array]:
    """Return one perfect matching from a one-factorization of K_n.

    Vertices 0,...,n-2 are treated modulo n-1 and vertex n-1 is the
    distinguished ``infinity`` vertex.  ``factor`` may be any integer in
    0,...,n-2.  Computing a requested factor directly avoids storing the full
    O(n^2) factorization when only K<<n matchings are used.
    """
    if n < 2 or n % 2:
        raise ValueError("n must be a positive even integer")
    if not 0 <= factor <= n - 2:
        raise ValueError("factor must lie in 0,...,n-2")
    modulus = n - 1
    half = (n - 2) // 2
    left = np.empty(n // 2, dtype=int)
    right = np.empty(n // 2, dtype=int)
    left[0], right[0] = n - 1, factor
    offsets = np.arange(1, half + 1, dtype=int)
    left[1:] = (factor + offsets) % modulus
    right[1:] = (factor - offsets) % modulus
    return left, right


def one_factorization(n: int) -> list[tuple[Array, Array]]:
    """Return all n-1 edge-disjoint perfect matchings of K_n."""
    return [one_factor_matching(n, a) for a in range(n - 1)]


def default_matchings(n: int) -> int:
    """Return K=2*ceil(log n), capped at the largest even K<n-1."""
    if n <= 4 or n % 2:
        raise ValueError("the cross-fitted procedure requires even n > 4")
    return min(2 * ceil(log(n)), n - 2)


def _as_vector_kernel(kernel: Array) -> tuple[Array, bool]:
    values = np.asarray(kernel, dtype=float)
    if values.ndim == 2:
        values = values[:, :, None]
        scalar = True
    elif values.ndim == 3:
        scalar = values.shape[2] == 1
    else:
        raise ValueError("kernel must have shape (n,n) or (n,n,q)")

    if values.shape[0] != values.shape[1]:
        raise ValueError("kernel must be square in its first two dimensions")
    if values.shape[0] <= 4 or values.shape[0] % 2:
        raise ValueError("the current implementation requires even n > 4")
    if values.shape[2] < 1:
        raise ValueError("kernel must contain at least one component")
    if not np.all(np.isfinite(values)):
        raise ValueError("kernel contains a non-finite value")
    if not np.allclose(values, np.swapaxes(values, 0, 1), rtol=1e-10, atol=1e-12):
        raise ValueError("kernel must be symmetric in its first two dimensions")
    return values, scalar


def project_psd(matrix: Array, *, floor: float = 0.0) -> Array:
    """Project a symmetric matrix onto the PSD cone in Frobenius norm."""
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, floor)
    return (eigenvectors * clipped) @ eigenvectors.T


def _normal_intervals(
    estimate: Array, standard_error: Array, confidence: float
) -> Array:
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie strictly between 0 and 1")
    critical = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return np.column_stack(
        (estimate - critical * standard_error, estimate + critical * standard_error)
    )


def balanced_inference_from_kernel(
    kernel: Array,
    *,
    k_matchings: int | None = None,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BalancedResult:
    """Estimate scalar or vector pairwise parameters with balanced pairs.

    Parameters
    ----------
    kernel:
        Symmetric array with shape ``(n,n)`` for a scalar parameter or
        ``(n,n,q)`` for q component parameters.  Diagonal entries are ignored.
    k_matchings:
        Positive even number K of edge-disjoint perfect matchings.  The default
        is ``2*ceil(log(n))``, capped at ``n-2``.
    confidence:
        Marginal normal confidence level.
    seed:
        Seed for random relabeling and random selection/order of the factors.
        The randomized factor design is the canonical released procedure; a
        fixed seed makes the selected graph exactly reproducible.
    """
    h, _ = _as_vector_kernel(kernel)
    n, _, q = h.shape
    k = default_matchings(n) if k_matchings is None else int(k_matchings)
    if k <= 0 or k % 2:
        raise ValueError("k_matchings must be a positive even integer")
    if k > n - 2:
        raise ValueError("for even n, an even k_matchings cannot exceed n-2")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    relabel = rng.permutation(n)
    factor_order = rng.permutation(n - 1)[:k]
    half = k // 2

    row_first = np.zeros((n, q), dtype=float)
    row_second = np.zeros((n, q), dtype=float)
    edge_values: list[Array] = []
    selected_edges: list[Array] = []

    for factor_position, factor_id in enumerate(factor_order):
        left0, right0 = one_factor_matching(n, int(factor_id))
        left = relabel[left0]
        right = relabel[right0]
        values = h[left, right, :]
        edge_values.append(values)
        selected_edges.append(np.column_stack((left, right)))
        target = row_first if factor_position < half else row_second
        target[left, :] += values
        target[right, :] += values

    y = np.vstack(edge_values)
    edges = np.vstack(selected_edges)
    m = y.shape[0]
    estimate = y.mean(axis=0)

    row_first /= half
    row_second /= half
    mean_first = row_first.mean(axis=0)
    mean_second = row_second.mean(axis=0)
    centered_first = row_first - mean_first
    centered_second = row_second - mean_second

    gamma1_raw = (
        centered_first.T @ centered_second
        + centered_second.T @ centered_first
    ) / (2.0 * (n - 4))

    centered_edges = y - estimate
    edge_covariance = centered_edges.T @ centered_edges / (m - 1)
    coefficient = 2.0 - 2.0 * (k - 1.0) / (m - 1.0)
    gamma2_raw = edge_covariance - coefficient * gamma1_raw

    # Form the unbiased raw estimators first, and only then impose the PSD
    # constraint.  Projection is useful numerically but creates upward bias at
    # the boundary; the manuscript reports this tradeoff explicitly.
    gamma1 = project_psd(gamma1_raw)
    gamma2 = project_psd(gamma2_raw)

    covariance = 4.0 * gamma1 / n + 2.0 * gamma2 / (n * k)
    covariance = 0.5 * (covariance + covariance.T)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    intervals = _normal_intervals(estimate, standard_error, confidence)

    row_projection = 0.5 * (row_first + row_second) - estimate
    return BalancedResult(
        estimate=estimate,
        covariance=covariance,
        standard_error=standard_error,
        confidence_interval=intervals,
        gamma1=gamma1,
        gamma2=gamma2,
        gamma1_raw=gamma1_raw,
        gamma2_raw=gamma2_raw,
        edge_covariance=edge_covariance,
        edge_covariance_coefficient=float(coefficient),
        row_projection=row_projection,
        half_projection_1=row_first - mean_first,
        half_projection_2=row_second - mean_second,
        selected_edges=edges,
        edge_values=y,
        factor_order=np.asarray(factor_order, dtype=int),
        relabeling=np.asarray(relabel, dtype=int),
        matchings=k,
        evaluated_pairs=m,
        confidence=confidence,
    )


def complete_inference_from_kernel(
    kernel: Array,
    *,
    confidence: float = 0.95,
) -> CompleteResult:
    """Regular complete-pair inference using only the first projection."""
    h, _ = _as_vector_kernel(kernel)
    n, _, _ = h.shape
    upper = np.triu_indices(n, k=1)
    y = h[upper[0], upper[1], :]
    estimate = y.mean(axis=0)

    diagonal = np.diagonal(h, axis1=0, axis2=1).T
    row_means = (h.sum(axis=1) - diagonal) / (n - 1)
    row_projection = row_means - estimate
    gamma1 = row_projection.T @ row_projection / (n - 1)
    covariance = 4.0 * gamma1 / n
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    intervals = _normal_intervals(estimate, standard_error, confidence)
    return CompleteResult(
        estimate=estimate,
        covariance=covariance,
        standard_error=standard_error,
        confidence_interval=intervals,
        gamma1=gamma1,
        row_projection=row_projection,
        evaluated_pairs=y.shape[0],
        confidence=confidence,
    )


def wasserstein_total_from_distances(
    distances: Array,
    *,
    k_matchings: int | None = None,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Balanced inference for V_W=E[W2(mu,nu)^2]/2."""
    d = np.asarray(distances, dtype=float)
    if np.any(d < -1e-12):
        raise ValueError("distances must be nonnegative")
    result = balanced_inference_from_kernel(
        0.5 * d * d,
        k_matchings=k_matchings,
        confidence=confidence,
        seed=seed,
    )
    output = result.scalar()
    output["attribution"] = 2.0 * output.pop("row_projection")
    output["half_attribution_1"] = 2.0 * output.pop("half_projection_1")
    output["half_attribution_2"] = 2.0 * output.pop("half_projection_2")
    return output


def transformed_summary_from_distances(
    distances: Array,
    transform: Callable[[Array], Array],
    *,
    k_matchings: int | None = None,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Balanced inference for D_psi=E[psi(W2(mu,nu))]."""
    d = np.asarray(distances, dtype=float)
    transformed = np.asarray(transform(d), dtype=float)
    if transformed.shape != d.shape:
        raise ValueError("transform must return an array with the same shape")
    return balanced_inference_from_kernel(
        transformed,
        k_matchings=k_matchings,
        confidence=confidence,
        seed=seed,
    ).scalar()


if __name__ == "__main__":
    rng = np.random.default_rng(20260717)
    x = rng.normal(size=200)
    distance_matrix = np.abs(x[:, None] - x[None, :])
    result = wasserstein_total_from_distances(distance_matrix, seed=20260717)
    print(
        f"estimate={result['estimate']:.6f}, "
        f"se={result['standard_error']:.6f}, "
        f"CI={result['confidence_interval']}, "
        f"Gamma1={result['gamma1']:.6f}, "
        f"Gamma2={result['gamma2']:.6f}, "
        f"pairs={result['evaluated_pairs']}"
    )
