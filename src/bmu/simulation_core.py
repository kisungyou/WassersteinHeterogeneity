"""Shared simulation helpers for the individual numerical experiments."""
from __future__ import annotations

from statistics import NormalDist

import numpy as np


def random_factor_orders(
    repetitions: int, n: int, k_matchings: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample an ordered K-subset of the n-1 one-factorization factors."""
    population = n - 1
    if repetitions * population <= 5_000_000:
        keys = rng.random((repetitions, population))
        return np.argsort(keys, axis=1)[:, :k_matchings]
    orders = np.empty((repetitions, k_matchings), dtype=int)
    for rep in range(repetitions):
        orders[rep] = rng.choice(population, size=k_matchings, replace=False)
    return orders


def balanced_scalar_from_edge_scores(
    x: np.ndarray,
    *,
    k_matchings: int,
    score_function,
    design_rng: np.random.Generator | None = None,
    factor_design: str = "randomized",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized balanced scalar inference for many Monte Carlo samples."""
    repetitions, n = x.shape
    if n % 2 or k_matchings % 2 or k_matchings > n - 2:
        raise ValueError("requires even n, even K, and K<=n-2")
    if factor_design not in {"randomized", "fixed"}:
        raise ValueError("factor_design must be 'randomized' or 'fixed'")
    if factor_design == "randomized":
        if design_rng is None:
            raise ValueError("design_rng is required for randomized factors")
        factor_orders = random_factor_orders(repetitions, n, k_matchings, design_rng)
    else:
        factor_orders = np.broadcast_to(
            np.arange(k_matchings, dtype=int), (repetitions, k_matchings)
        )

    first = np.zeros((repetitions, n), dtype=float)
    second = np.zeros((repetitions, n), dtype=float)
    edge_sum = np.zeros(repetitions, dtype=float)
    edge_sq_sum = np.zeros(repetitions, dtype=float)
    half = k_matchings // 2
    row_ids = np.arange(repetitions)[:, None]
    offsets = np.arange(1, n // 2, dtype=int)[None, :]
    modulus = n - 1

    for position in range(k_matchings):
        factor = factor_orders[:, position]
        left = np.empty((repetitions, n // 2), dtype=int)
        right = np.empty_like(left)
        left[:, 0] = n - 1
        right[:, 0] = factor
        left[:, 1:] = (factor[:, None] + offsets) % modulus
        right[:, 1:] = (factor[:, None] - offsets) % modulus

        scores = np.asarray(
            score_function(x[row_ids, left], x[row_ids, right]), dtype=float
        )
        edge_sum += scores.sum(axis=1)
        edge_sq_sum += (scores * scores).sum(axis=1)
        target = first if position < half else second
        np.add.at(target, (row_ids, left), scores)
        np.add.at(target, (row_ids, right), scores)

    m_edges = k_matchings * n / 2
    estimate = edge_sum / m_edges
    row_first = first / half
    row_second = second / half
    mean_first = row_first.mean(axis=1)
    mean_second = row_second.mean(axis=1)

    gamma1_raw = np.sum(
        (row_first - mean_first[:, None])
        * (row_second - mean_second[:, None]),
        axis=1,
    ) / (n - 4)
    gamma1 = np.maximum(gamma1_raw, 0.0)

    edge_variance = (edge_sq_sum - m_edges * estimate * estimate) / (m_edges - 1)
    coefficient = 2.0 - 2.0 * (k_matchings - 1.0) / (m_edges - 1.0)
    gamma2_raw = edge_variance - coefficient * gamma1_raw
    gamma2 = np.maximum(gamma2_raw, 0.0)
    variance = 4.0 * gamma1 / n + 2.0 * gamma2 / (n * k_matchings)
    return estimate, variance, gamma1, gamma2, gamma1_raw


def near_degenerate_setting(
    *,
    n: int,
    k_matchings: int,
    p: float,
    repetitions: int,
    seed: int,
    factor_design: str = "randomized",
    design_seed: int | None = None,
) -> dict[str, float | str]:
    """Two-point Gaussian population with h=W2^2/2."""
    data_rng = np.random.default_rng(seed)
    design_rng = np.random.default_rng(
        seed + 1_000_003 if design_seed is None else design_seed
    )
    x = data_rng.random((repetitions, n)) < p
    estimate, variance, gamma1, gamma2, gamma1_raw = balanced_scalar_from_edge_scores(
        x,
        k_matchings=k_matchings,
        score_function=lambda a, b: 2.0 * (a != b),
        design_rng=design_rng,
        factor_design=factor_design,
    )
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    critical = NormalDist().inv_cdf(0.975)
    theta = 4.0 * p * (1.0 - p)
    lower = estimate - critical * standard_error
    upper = estimate + critical * standard_error
    coverage = np.mean((lower <= theta) & (theta <= upper))

    count = x.sum(axis=1)
    complete = 4.0 * count * (n - count) / (n * (n - 1))
    g_plus = np.where(count > 0, 2.0 * (n - count) / (n - 1) - complete, 0.0)
    g_minus = np.where(count < n, 2.0 * count / (n - 1) - complete, 0.0)
    complete_gamma1 = (
        count * g_plus**2 + (n - count) * g_minus**2
    ) / (n - 1)
    complete_se = 2.0 * np.sqrt(complete_gamma1 / n)
    complete_lower = complete - critical * complete_se
    complete_upper = complete + critical * complete_se
    complete_coverage = np.mean(
        (complete_lower <= theta) & (theta <= complete_upper)
    )

    gamma1_true = 4.0 * p * (1.0 - p) * (2.0 * p - 1.0) ** 2
    gamma2_true = 16.0 * p**2 * (1.0 - p) ** 2
    oracle_variance = 4.0 * gamma1_true / n + 2.0 * gamma2_true / (n * k_matchings)
    mean_reported_variance = float(np.mean(variance))
    mean_length = float(np.mean(upper - lower))
    return {
        "model": "two_point",
        "regime": "exact boundary" if p == 0.5 else "near boundary",
        "factor_design": factor_design,
        "p": p,
        "n": n,
        "K": k_matchings,
        "K2_over_sqrt_n": k_matchings**2 / np.sqrt(n),
        "BE_design_factor_p3": float(
            75.0 * (2.0 * k_matchings - 1.0) ** 10
            * np.sqrt(2.0 / (n * k_matchings))
        ),
        "target_VW": theta,
        "balanced_coverage": float(coverage),
        "complete_wald_coverage": float(complete_coverage),
        "balanced_mean_length": mean_length,
        "scaled_balanced_mean_length": float(mean_length * np.sqrt(n * k_matchings)),
        "complete_wald_mean_length": float(np.mean(complete_upper - complete_lower)),
        "balanced_mean_estimate": float(np.mean(estimate)),
        "balanced_mean_standard_error": float(np.mean(standard_error)),
        "balanced_mean_variance": mean_reported_variance,
        "oracle_variance": float(oracle_variance),
        "variance_inflation_percent": float(
            100.0 * (mean_reported_variance / oracle_variance - 1.0)
        ),
        "mean_gamma1": float(np.mean(gamma1)),
        "mean_gamma1_raw": float(np.mean(gamma1_raw)),
        "sd_gamma1_raw": float(np.std(gamma1_raw, ddof=1)),
        "mean_gamma2": float(np.mean(gamma2)),
        "repetitions": repetitions,
    }


def normal_kernel_setting(
    *,
    n: int,
    k_matchings: int,
    repetitions: int,
    seed: int,
    factor_design: str = "randomized",
    design_seed: int | None = None,
) -> dict[str, float | str]:
    """Unbounded regular kernel: X~N(0,1), h=(X-X')^2/2."""
    data_rng = np.random.default_rng(seed)
    design_rng = np.random.default_rng(
        seed + 1_000_003 if design_seed is None else design_seed
    )
    x = data_rng.normal(size=(repetitions, n))
    estimate, variance, gamma1, gamma2, gamma1_raw = balanced_scalar_from_edge_scores(
        x,
        k_matchings=k_matchings,
        score_function=lambda a, b: 0.5 * (a - b) ** 2,
        design_rng=design_rng,
        factor_design=factor_design,
    )
    se = np.sqrt(np.maximum(variance, 0.0))
    critical = NormalDist().inv_cdf(0.975)
    target = 1.0
    coverage = np.mean(
        (estimate - critical * se <= target)
        & (target <= estimate + critical * se)
    )
    oracle_variance = 4.0 * 0.5 / n + 2.0 * 1.0 / (n * k_matchings)
    mean_reported_variance = float(np.mean(variance))
    mean_length = float(np.mean(2.0 * critical * se))
    return {
        "model": "Gaussian squared difference",
        "regime": "regular, unbounded",
        "factor_design": factor_design,
        "p": np.nan,
        "n": n,
        "K": k_matchings,
        "K2_over_sqrt_n": k_matchings**2 / np.sqrt(n),
        "BE_design_factor_p3": float(
            75.0 * (2.0 * k_matchings - 1.0) ** 10
            * np.sqrt(2.0 / (n * k_matchings))
        ),
        "target_VW": target,
        "balanced_coverage": float(coverage),
        "complete_wald_coverage": np.nan,
        "balanced_mean_length": mean_length,
        "scaled_balanced_mean_length": float(mean_length * np.sqrt(n * k_matchings)),
        "complete_wald_mean_length": np.nan,
        "balanced_mean_estimate": float(np.mean(estimate)),
        "balanced_mean_standard_error": float(np.mean(se)),
        "balanced_mean_variance": mean_reported_variance,
        "oracle_variance": float(oracle_variance),
        "variance_inflation_percent": float(
            100.0 * (mean_reported_variance / oracle_variance - 1.0)
        ),
        "mean_gamma1": float(np.mean(gamma1)),
        "mean_gamma1_raw": float(np.mean(gamma1_raw)),
        "sd_gamma1_raw": float(np.std(gamma1_raw, ddof=1)),
        "mean_gamma2": float(np.mean(gamma2)),
        "repetitions": repetitions,
    }


def complete_component_estimates(values: np.ndarray) -> tuple[float, float]:
    """Metric variance and regular first-projection standard error."""
    n = values.shape[0]
    center = values.mean(axis=0)
    centered_sqnorm = np.sum((values - center) ** 2, axis=1)
    estimate = float(centered_sqnorm.sum() / (n - 1))
    attribution = n * centered_sqnorm / (n - 1) - estimate
    sigma_a2 = float(np.sum(attribution**2) / (n - 1))
    return estimate, float(np.sqrt(sigma_a2 / n))


def metric_variance_of_rows(q: np.ndarray) -> float:
    """Metric variance of discretized Hilbert/quantile rows."""
    n = q.shape[0]
    qbar = q.mean(axis=0)
    return float(np.sum((q - qbar) ** 2) / ((n - 1) * q.shape[1]))
