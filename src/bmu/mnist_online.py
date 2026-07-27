#!/usr/bin/env python3
"""Regenerate the MNIST uncertainty analysis from the public distance files.

The original within-digit 500x500 Wasserstein distance matrices are archived in
Kisung You's public repository.  This script downloads those matrices, reads the
RData object ``pdmat``, and computes both complete-pair regular inference and
the balanced cross-fitted inference developed in the manuscript.

Usage
-----
    python -m bmu.mnist_online --outdir mnist_recomputed

Requirements
------------
    numpy, pandas, requests, pyreadr

The script is intentionally separate from the offline simulation script because
it needs an internet connection and downloads approximately 20 MB of archived
matrices.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

try:
    import pyreadr
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "pyreadr is required. Install it with `python -m pip install pyreadr`."
    ) from exc

from .balanced_uncertainty import (
    complete_inference_from_kernel,
    wasserstein_total_from_distances,
)

BASE_URL = (
    "https://raw.githubusercontent.com/kisungyou/papers/master/"
    "06-Wasserstein-Heterogeneity/code/data/distances"
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    handle.write(block)


def read_pdmat(path: Path) -> np.ndarray:
    objects = pyreadr.read_r(str(path))
    if "pdmat" not in objects:
        names = ", ".join(objects.keys())
        raise KeyError(f"{path.name} does not contain pdmat; found: {names}")
    matrix = np.asarray(objects["pdmat"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"pdmat in {path.name} is not square: {matrix.shape}")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
        raise ValueError(f"pdmat in {path.name} is not symmetric")
    return matrix


def omnibus_equal_totals(estimates: np.ndarray, variances: np.ndarray) -> tuple[float, int]:
    """Wald test of equality of ten independent scalar totals."""
    groups = estimates.size
    contrast = np.column_stack((-np.ones(groups - 1), np.eye(groups - 1)))
    covariance = np.diag(variances)
    difference = contrast @ estimates
    middle = contrast @ covariance @ contrast.T
    statistic = float(difference @ np.linalg.solve(middle, difference))
    return statistic, groups - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("mnist_recomputed"))
    parser.add_argument("--K", type=int, default=14, help="even number of matchings")
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data_dir = args.outdir / "distance_matrices"

    rows: list[dict[str, float | int]] = []
    for digit in range(10):
        path = data_dir / f"digit_{digit}.RData"
        download(f"{BASE_URL}/digit_{digit}.RData", path)
        distances = read_pdmat(path)
        kernel = 0.5 * distances**2

        # CompleteResult is vector-valued; extract its scalar entries directly.
        complete_result = complete_inference_from_kernel(kernel)
        balanced = wasserstein_total_from_distances(
            distances,
            k_matchings=args.K,
            confidence=0.95,
            seed=args.seed + digit,
        )
        rows.append(
            {
                "digit": digit,
                "n": distances.shape[0],
                "complete_estimate_VW": float(complete_result.estimate[0]),
                "complete_se_regular": float(complete_result.standard_error[0]),
                "complete_lower_regular": float(complete_result.confidence_interval[0, 0]),
                "complete_upper_regular": float(complete_result.confidence_interval[0, 1]),
                "complete_gamma1": float(complete_result.gamma1[0, 0]),
                "balanced_estimate_VW": balanced["estimate"],
                "balanced_se": balanced["standard_error"],
                "balanced_lower": balanced["confidence_interval"][0],
                "balanced_upper": balanced["confidence_interval"][1],
                "balanced_gamma1": balanced["gamma1"],
                "balanced_gamma1_raw": balanced["gamma1_raw"],
                "balanced_gamma2": balanced["gamma2"],
                "balanced_gamma2_raw": balanced["gamma2_raw"],
                "K": args.K,
                "evaluated_pairs": balanced["evaluated_pairs"],
            }
        )
        print(f"digit {digit}: complete={rows[-1]['complete_estimate_VW']:.6f}, "
              f"balanced={rows[-1]['balanced_estimate_VW']:.6f}")

    output = pd.DataFrame(rows)
    estimates = output["balanced_estimate_VW"].to_numpy()
    variances = output["balanced_se"].to_numpy() ** 2
    q, df = omnibus_equal_totals(estimates, variances)
    output["balanced_omnibus_Q"] = q
    output["balanced_omnibus_df"] = df
    output.to_csv(args.outdir / "mnist_balanced_results.csv", index=False)
    print(f"Balanced omnibus equality test: Q={q:.6f}, df={df}")
    print(f"Results written to {args.outdir / 'mnist_balanced_results.csv'}")


if __name__ == "__main__":
    main()
