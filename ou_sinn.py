"""Reproduce the SINN paper's Ornstein-Uhlenbeck statistics benchmark."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr
import torch
from torch import nn

DT = 0.10
DECAY, SIGMA = 1.0, 0.50
VARIANCE = SIGMA**2 / (2*DECAY)
BANDWIDTH = 0.08
LAGS = (1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30)
ROOT = Path(__file__).resolve().parent


class SINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = nn.LSTM(1, 1, num_layers=2, batch_first=True)
        self.readout = nn.Linear(1, 1, bias=False)
        # Start with nonzero noise transmission and different memory lengths.
        with torch.no_grad():
            self.readout.weight.fill_(2.0)
            for layer, forget_bias in enumerate((-1.4, 2.0)):
                input_weights = getattr(self.recurrent, f"weight_ih_l{layer}")
                hidden_weights = getattr(self.recurrent, f"weight_hh_l{layer}")
                input_bias = getattr(self.recurrent, f"bias_ih_l{layer}")
                hidden_bias = getattr(self.recurrent, f"bias_hh_l{layer}")
                input_weights.mul_(0.10)
                input_weights[2, 0] = 0.8
                hidden_weights.mul_(0.10)
                input_bias.zero_()
                hidden_bias.zero_()
                input_bias[1] = forget_bias
                input_bias[0] = 0.8
                input_bias[3] = 0.8

    def forward(self, noise):
        hidden, _ = self.recurrent(noise)
        return self.readout(hidden).squeeze(-1)


def normal_pdf(x, variance):
    return torch.exp(-x.square() / (2*variance)) / np.sqrt(2*np.pi*variance)


def statistics(series, grid, lags):
    centered = series - series.mean()
    acf = torch.stack([(centered[:, lag:] * centered[:, :-lag]).mean()
                       for lag in lags]) / centered.square().mean().clamp_min(1e-10)
    samples = series.flatten()
    density = normal_pdf(grid[:, None] - samples, BANDWIDTH**2).mean(dim=1)
    return acf, density


def mixed_loss(estimate, target):
    error = estimate - target
    return error.abs().mean() + error.square().mean()


def score(series, grid, lags):
    acf, density = statistics(series, grid, lags)
    true_acf = torch.exp(-DECAY * DT * torch.tensor(lags))
    # Compare KDEs at the same effective bandwidth by convolving the target.
    true_density = normal_pdf(grid, VARIANCE + BANDWIDTH**2)
    return acf, density, true_acf, true_density


def evaluate(model, grid, eval_noise):
    with torch.no_grad():
        values = model(eval_noise)[:, 64:]
        acf, density, true_acf, true_density = score(values, grid, LAGS)
        sorted_values = np.sort(values.numpy().ravel())
        n = len(sorted_values)
        target_cdf = ndtr(sorted_values / np.sqrt(VARIANCE))
        ks = max(np.max(np.arange(1, n + 1) / n - target_cdf),
                 np.max(target_cdf - np.arange(n) / n))
        metrics = {
            "stationary_mean": float(values.mean()),
            "stationary_sd": float(values.std(unbiased=False)),
            "acf_rmse": float((acf - true_acf).square().mean().sqrt()),
            "pdf_rmse": float((density - true_density).square().mean().sqrt()),
            "normal_cdf_max_error": float(ks),
        }
    return metrics, acf.numpy(), density.numpy()


def reference_paths(seed, count, length):
    """Sample exact discrete OU transitions from the stationary initial law."""
    rng = np.random.default_rng(seed)
    rho = np.exp(-DECAY*DT)
    series = np.empty((count, length))
    series[:, 0] = rng.normal(scale=np.sqrt(VARIANCE), size=count)
    noise = rng.normal(scale=np.sqrt(VARIANCE * (1 - rho**2)), size=(length - 1, count))
    for step in range(1, length):
        series[:, step] = rho * series[:, step - 1] + noise[step - 1]
    return series


def run(data, output, steps):
    torch.set_num_threads(1)
    torch.manual_seed(314)
    train_rng = torch.Generator().manual_seed(316)
    eval_rng = torch.Generator().manual_seed(318)
    grid = torch.linspace(-1.2, 1.2, 49)
    eval_noise = torch.randn((512, 256, 1), generator=eval_rng)
    model = SINN()
    initial, _, _ = evaluate(model, grid, eval_noise)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for step in range(steps):
        values = model(torch.randn((24, 160, 1), generator=train_rng))[:, 40:]
        acf, density, true_acf, true_density = score(values, grid, LAGS)
        loss = mixed_loss(acf, true_acf) + mixed_loss(density, true_density)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0:
            print(f"OU step {step + 1}: loss {loss.item():.4f}", flush=True)

    fitted, acf, density = evaluate(model, grid, eval_noise)
    metrics = {"train_steps": steps, "target_sd": float(np.sqrt(VARIANCE)),
               "evaluation_trajectories": len(eval_noise),
               "initial": initial, "fitted": fitted}
    data.mkdir(parents=True, exist_ok=True)
    np.savetxt(data / "ou_reference_paths.csv.gz", reference_paths(2026, 128, 192),
               delimiter=",", fmt="%.8f",
               header=",".join(f"t{t:03d}" for t in range(192)), comments="")
    output.mkdir(parents=True, exist_ok=True)
    (output / "ou_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45), layout="constrained")
    axes[0].plot(grid, normal_pdf(grid, VARIANCE + BANDWIDTH**2), color="#222222",
                 lw=1.5, label="Analytic target")
    axes[0].plot(grid, density, "--", color="#087f8c", lw=2, label="Fitted SINN")
    axes[0].set(xlabel="OU state", ylabel="Smoothed density")
    axes[0].legend(frameon=False, fontsize=8)
    t = np.array(LAGS) * DT
    axes[1].plot(t, np.exp(-DECAY*t), color="#222222", lw=1.5)
    axes[1].plot(t, acf, "o--", color="#087f8c", ms=3, lw=1.5)
    axes[1].set(xlabel="Lag (time units)", ylabel="Autocorrelation")
    fig.savefig(output / "ou_check.png", dpi=200)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    run(args.data, args.out, args.steps)
