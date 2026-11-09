# Discovering hidden capacity-correlation groups

Can buildings with the same fragility curve still fall into groups that tend to be damaged together? This project reproduces the Ornstein–Uhlenbeck (OU) benchmark in Section 4.2.1 of [Zhu, Tang, and Kim (2023)](https://doi.org/10.1016/j.jcp.2022.111819) and extends their statistics-informed learning approach to paired binary building damage. The [paper's original implementation](https://github.com/SINN-model/SINN) includes the OU example. The two implementations here were written in PyTorch from the paper and the project model.

The OU benchmark uses two one-unit LSTM layers to transform white noise into trajectories. The training loss matches the stationary density and autocorrelation. For the extension, a two-hidden-layer MLP infers group memberships and shared-variance weights from simulated PGA and damage. A Gaussian factor model fixes the marginal fragility by construction. The classifier uses a pairwise composite likelihood.

## Reproduce

Tested with Python 3.12 and PyTorch 2.15.0 CPU. From the repository root, create and activate a virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python ou_sinn.py
python capacity.py
python -m unittest discover -s tests -v
```

On a CPU-only Linux machine, you can install the smaller CPU build first with `python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`. The next install will use that build.

Use `python capacity.py --generate` to regenerate the building dataset with seed 42 and refit the models. Running `ou_sinn.py` also regenerates the exact OU reference paths using seed 2026. It fits against analytic statistics, so the reference paths are available for independent checks. Both scripts accept `--data`, `--out`, and `--steps`. Use a separate output directory if you want to preserve the reported results.

## Data

All observations are simulated. The building dataset uses 120 fixed profiles, with capacity and intensity redrawn for each independent portfolio.

| File | Contents |
|---|---|
| `data/buildings.csv` | One row per building profile with features `x1` and `x2` and a zero-based building ID |
| `data/portfolios.csv` | 12,000 rows with portfolio ID, split, building ID, PGA in g, and binary damage |
| `data/group_truth.csv` | One-based simulation group labels, used only after fitting |
| `data/ou_reference_paths.csv.gz` | 128 independent stationary OU paths, each with 192 values at 0.1 time-unit spacing |

For the damage benchmark, portfolios 0–79 are used for training and 80–99 for testing. The building profiles are shared across both splits. The two weight parameters are 0.60 and 0.30 in the simulator. The common fragility median and log dispersion are 0.30 g and 0.60.

## Results

| Model | Held-out pairwise negative log likelihood |
|---|---:|
| Independent capacities | 1.06354 |
| One fitted correlation | 1.06013 |
| Two-group MLP | 1.05243 |
| Oracle | 1.05221 |

After label alignment, the MLP recovered 120 of 120 assignments and estimated weights of 0.610 and 0.326. The test split evaluates new portfolios on the same building profiles. The bootstrap resamples complete test portfolios and holds the fitted models fixed.

The OU reproduction reduced normalized autocorrelation RMSE from 0.061 to 0.013 and smoothed density RMSE from 0.203 to 0.038 on fresh trajectories. See `results/` for detailed metrics, membership probabilities, and plots.
