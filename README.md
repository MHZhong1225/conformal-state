# Conformal Time Series

Code for **Dynamic State-Space Conformal Control (DSS-CI)**, an online conformal calibration method for time-series prediction intervals under nonstationarity.

DSS-CI decomposes the conformal threshold into:

- a state-space feedforward term that predicts the future scale of nonconformity scores;
- a coverage-feedback correction term that maintains calibration;
- an innovation-based gain schedule that increases the feedback step size when the score dynamics deviate from the Kalman prediction.

The repository contains the method implementation, baseline conformal controllers, experiment configs, saved results, and plotting scripts used for the method paper.

## Repository Layout

```text
.
|-- core/
|   |-- methods.py          # Baselines and DSS-CI experiment implementation
|   |-- state_space.py      # 1D Kalman filter and numerical helpers
|   |-- model_scores.py     # Forecast generation with Darts models
|   |-- quantile.py         # Weighted conformal utilities
|   `-- synthetic_scores.py # Synthetic score generators
|-- tests/
|   |-- base_test.py        # Main experiment runner
|   |-- base_plots.py       # Coverage/grid/set-size plots
|   |-- inset_plot.py       # Pairwise comparison plots and tables
|   |-- configs/            # YAML experiment configurations
|   |-- datasets/           # Data used by the experiment runner
|   |-- results/            # Pickled experiment outputs
|   `-- plots/              # Generated figures and LaTeX tables
|-- css.py                  # Standalone DSS-CI class-style prototype
|-- requirements.txt
`-- README.md
```

## Installation

The experiments use Python with NumPy/Pandas/Statsmodels/Darts and plotting libraries.

```bash
git clone <repo-url>
cd conformal-time-series

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Some forecast models, especially `TransformerModel` from Darts, may require PyTorch. If your Darts installation does not pull in a suitable PyTorch build automatically, install PyTorch for your platform before running Transformer experiments.

## Method Implementation

The main experiment-facing DSS-CI function is:

```python
from core import dss_cc

result = dss_cc(
    scores=scores,
    alpha=0.1,
    lr=0.1,
    ahead=1,
    T_burnin=100,
    eta_min=0.05,
    eta_max=0.5,
    q_max=2000,
)

q = result["q"]
```

The returned dictionary includes:

- `q`: conformal thresholds used over time;
- `g`: state-space feedforward thresholds;
- `z`: feedback correction states;
- `eta`: dynamic feedback step sizes;
- `kappa`: normalized Kalman innovation values;
- `covered`: online coverage indicators.

For a more object-oriented reference implementation, see `css.py`.

## Running Experiments

Run commands from the `tests/` directory.

### Single Experiment

```bash
cd tests
python3 base_test.py configs/GOOGL.yaml
```

This writes or updates:

```text
tests/results/GOOGL.pkl
```

To rerun only selected methods, pass a comma-separated overwrite list:

```bash
python3 base_test.py configs/GOOGL.yaml DSS-CI
python3 base_test.py configs/GOOGL.yaml "Quantile,DSS-CI"
```

### All Configured Experiments

```bash
cd tests
bash run_tests.sh
```

`run_tests.sh` launches one background process per YAML config. This can be CPU/GPU intensive, especially for configs that include Transformer forecasts. For debugging or partial reproduction, run one config at a time.

## Experiment Configs

Experiment settings live in `tests/configs/*.yaml`. Each config defines:

- dataset and forecasting model(s);
- score function, such as `signed-residual` or `cqr-asymmetric`;
- target miscoverage `alpha`;
- burn-in length `T_burnin`;
- learning-rate grids for each method;
- DSS-CI parameters, including `eta_min`, `eta_max`, `q_max`, and Kalman parameters.

Representative configs:

- `AMZN.yaml`, `GOOGL.yaml`, `MSFT.yaml`: stock-price experiments;
- `elec2.yaml`: electricity demand;
- `daily-climate.yaml`: daily climate series;
- `tx-COVID-deaths-4wk.yaml` and related state configs: four-week COVID-19 death forecasts;
- `stationary.yaml`, `increasing_lownoise.yaml`, `mix1.yaml`: synthetic score processes.

For asymmetric score functions, `base_test.py` runs separate lower- and upper-side calibrations, using `alpha / 2` for each side.

## Plotting

After results have been generated, create grid plots and set-size summaries:

```bash
cd tests
bash make_plots.sh
```

The plotting scripts write figures and LaTeX tables under:

```text
tests/plots/
tests/plots/1v1/
```

You can also run a single pairwise plot manually, for example:

```bash
python3 inset_plot.py \
  --filename results/GOOGL.pkl \
  --key1 Quantile --lr1 0.1 \
  --key2 DSS-CI --lr2 0.1 \
  --window_length 100 \
  --window_start 2300 \
  --window_loc "upper left" \
  --coverage_average_length 50 \
  --set_inset
```

## Data

The experiment loader is `tests/datasets.py`. It expects the prepared datasets to be available under `tests/datasets/`, including stock, electricity, climate, and processed COVID-19 forecast files.

Large raw datasets are not regenerated by the runner. If a dataset file is missing, place the processed file in the expected path or update `tests/datasets.py` to point to the new location.

## Baselines

The experiment runner currently includes:

- trailing-window empirical quantile;
- Adaptive Conformal Inference (ACI);
- clipped ACI;
- quantile plus integrator plus scorecaster, labeled as conformal PID in plots;
- CPTC;
- DSS-CI.

Most baselines and DSS-CI are implemented in `core/methods.py`.

## Notes on Reproducibility

- Forecasts and scorecaster outputs may be cached under `tests/datasets/proc/` and `tests/.cache/scorecaster/`.
- Existing pickled results under `tests/results/` are reused unless a method name is passed in the overwrite list.
- Transformer results may vary across hardware, PyTorch versions, and random initialization.
- Some scripts use `python`; if your environment only exposes `python3`, either invoke the scripts with `python3` directly or create a `python` alias inside your virtual environment.

