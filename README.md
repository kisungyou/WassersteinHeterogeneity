# Between-measure uncertainty

This package reproduces the numerical examples in *Quantifying between-measure
uncertainty in Wasserstein space*. 

The examples study metric variance among probability measures. They also
demonstrate balanced incomplete-pair inference, behavior near degeneracy, and
error caused by reconstructing each measure from finite data.

## Quick start

Create a fresh environment with Python 3.10 or newer. Then run:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Open the notebooks in numerical order. Each notebook finds the package root
whether Jupyter starts here or inside `notebooks`.

## Notebook map

| Notebook | Manuscript example |
|---|---|
| `01_gaussian_decomposition.ipynb` | Mean and covariance components for diagonal Gaussian measures |
| `02_balanced_inference.ipynb` | Coverage near first-order degeneracy and studentization diagnostics |
| `03_measure_reconstruction.ipynb` | Error from empirical reconstruction of latent measures |
| `04_mnist_application.ipynb` | Between-digit uncertainty summaries for the MNIST application |

Every notebook runs offline with deterministic seeds. Lightweight simulations
provide a quick check, while the files in `data` preserve the paper-scale
results used in the manuscript.

Set `PAPER_SCALE = True` in the first three notebooks to rerun the full
simulation settings. Those runs take longer because they use the manuscript's
sample sizes and repetition counts.

## MNIST data

The default MNIST notebook uses the archived summary in
`data/mnist_precomputed_results.csv`. This route is fully offline and
reproduces the reported complete-pair estimates and regular intervals.

Exact recomputation from the original distance matrices requires network
access and two optional packages:

```sh
python -m pip install -r requirements-online.txt
PYTHONPATH=src python -m bmu.mnist_online --outdir mnist_recomputed
```

The online script downloads the public distance matrices from the author's
earlier repository. It then computes complete-pair and balanced intervals from
the matrices themselves.

## Package layout

- `src/bmu` contains the released balanced inference implementation and shared
  simulation helpers.
- `data` contains paper-scale numerical results and the offline MNIST summary.
- `notebooks` contains the executable examples.

The research scripts used for the full article remain separate from this
public tutorial package. The notebooks state when they display archived
paper-scale results and when they perform a fresh simulation.
