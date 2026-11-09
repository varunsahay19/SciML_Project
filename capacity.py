"""Fit and evaluate capacity correlation using only attributes, PGA, and damage."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
N_BUILDINGS, N_PORTFOLIOS, N_TRAIN = 120, 100, 80
THETA, BETA = 0.30, 0.60
TRUE_WEIGHTS = np.array([0.60, 0.30])
SEED = 42
PAIR_I, PAIR_J = np.triu_indices(N_BUILDINGS, k=1)
nodes, weights = np.polynomial.legendre.leggauss(12)
NODES, QUAD_WEIGHTS = (nodes + 1) / 2, weights / 2


def generate(directory):
    """Generate the outline's independent synthetic portfolios and save CSVs."""
    rng = np.random.default_rng(SEED)
    features = rng.uniform(-1, 1, (N_BUILDINGS, 2))
    groups = (features[:, 0] * features[:, 1] <= 0).astype(int)
    pga = rng.uniform(0.15, 0.60, (N_PORTFOLIOS, N_BUILDINGS))
    shared = rng.normal(size=(N_PORTFOLIOS, 2))
    residual = rng.normal(size=pga.shape)
    w = TRUE_WEIGHTS[groups]
    log_capacity = np.log(THETA) + BETA * (
        np.sqrt(w) * shared[:, groups] + np.sqrt(1 - w) * residual
    )
    damage = (np.log(pga) >= log_capacity).astype(int)
    directory.mkdir(parents=True, exist_ok=True)
    files = (
        ("buildings.csv", ("building_id", "x1", "x2"),
         ((i, *features[i]) for i in range(N_BUILDINGS))),
        ("portfolios.csv", ("portfolio_id", "split", "building_id", "pga_g", "damage"),
         ((r, "train" if r < N_TRAIN else "test", i, pga[r, i], damage[r, i])
          for r in range(N_PORTFOLIOS) for i in range(N_BUILDINGS))),
        ("group_truth.csv", ("building_id", "true_group"),
         ((i, groups[i] + 1) for i in range(N_BUILDINGS))),
    )
    for name, header, rows in files:
        with (directory / name).open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)


def observed_pair_arrays(directory):
    features = np.genfromtxt(directory / "buildings.csv", delimiter=",", names=True)
    rows = np.genfromtxt(directory / "portfolios.csv", delimiter=",", names=True,
                         dtype=None, encoding="utf-8")
    rows.sort(order=["portfolio_id", "building_id"])
    features.sort(order="building_id")
    test = (rows["split"] == "test").reshape(N_PORTFOLIOS, N_BUILDINGS)
    if np.any(test != test[:, :1]) or np.sum(~test[:, 0]) != N_TRAIN:
        raise ValueError("The split must keep each portfolio intact.")
    x = np.column_stack([features["x1"], features["x2"]])
    z = np.log(rows["pga_g"].reshape(test.shape) / THETA) / BETA
    sign = 2 * rows["damage"].reshape(test.shape) - 1
    a = (sign * z)[:, PAIR_I]
    b = (sign * z)[:, PAIR_J]
    sign_pair = sign[:, PAIR_I] * sign[:, PAIR_J]
    return x, np.stack((a, b, sign_pair), axis=-1), test[:, 0]


def joint_probability(a, b, rho):
    """Bivariate normal CDF via integration of the density over correlation."""
    a, b, rho = np.broadcast_arrays(a, b, rho)
    t = rho[..., None] * NODES
    variance = 1 - t**2
    density = np.exp(-(a[..., None]**2 - 2*t*a[..., None]*b[..., None]
                       + b[..., None]**2) / (2*variance))
    return ndtr(a) * ndtr(b) + rho * np.sum(
        QUAD_WEIGHTS * density / (2*np.pi*np.sqrt(variance)), axis=-1
    )


def torch_joint_probability(a, b, rho):
    nodes = torch.as_tensor(NODES, dtype=a.dtype, device=a.device)
    weights = torch.as_tensor(QUAD_WEIGHTS, dtype=a.dtype, device=a.device)
    t = rho[:, None] * nodes
    variance = 1 - t.square()
    a_col, b_col = a[:, None], b[:, None]
    density = torch.exp(-(a_col.square() - 2*t*a_col*b_col + b_col.square())
                        / (2*variance))
    correction = rho * ((density / variance.sqrt()) @ weights) / (2*np.pi)
    return torch.special.ndtr(a) * torch.special.ndtr(b) + correction


class GroupMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2, 16), nn.Tanh(),
                                  nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 2))
        self.weight_logits = nn.Parameter(torch.tensor([0.0, -0.4]))

    def forward(self, x):
        return self.gate(x).softmax(dim=-1), 0.95 * self.weight_logits.sigmoid()


def portfolio_losses(observed, rho):
    a, b, sign = np.moveaxis(observed, -1, 0)
    prob = joint_probability(a, b, sign * rho)
    return -np.log(np.maximum(prob, 1e-12)).mean(axis=1)


def train_model(x, observed, steps, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = GroupMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    profiles = torch.tensor(x, dtype=torch.float32)
    pairs = torch.tensor(observed, dtype=torch.float32)
    pair_i, pair_j = torch.tensor(PAIR_I), torch.tensor(PAIR_J)
    for _ in range(steps):
        r = rng.integers(len(observed), size=4096)
        pair = rng.integers(len(PAIR_I), size=4096)
        a, b, sign = pairs[r, pair].unbind(-1)
        q, w = model(profiles)
        rho = (q[pair_i[pair]] * q[pair_j[pair]] * w).sum(dim=1)
        prob = torch_joint_probability(a, b, sign * rho)
        loss = -prob.clamp_min(1e-8).log().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        q, w = model(profiles)
    return q.numpy(), w.numpy()


def pair_correlations(q, w):
    return (q[PAIR_I] * q[PAIR_J] * w).sum(axis=1).astype(float)


def align_groups(q, w, truth):
    """Use truth only after fitting to resolve arbitrary output labels."""
    if np.mean(q.argmax(axis=1) == truth) < 0.5:
        return q[:, ::-1], w[::-1]
    return q, w


def plot_groups(x, truth, q, output):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharex=True, sharey=True,
                             layout="constrained")
    for ax, label, values in zip(axes, ["Simulated groups", "MLP predictions"],
                                 [truth, q.argmax(axis=1)]):
        for k, color, marker in [(0, "#087f8c", "o"), (1, "#a64425", "^")]:
            mask = values == k
            ax.scatter(*x[mask].T, c=color, marker=marker, s=22, label=f"Group {k+1}")
        ax.axvline(0, color="#aaaaaa", lw=0.8, zorder=0)
        ax.axhline(0, color="#aaaaaa", lw=0.8, zorder=0)
        ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05),
               xlabel=r"$x_1$", title=label, aspect="equal")
    axes[0].set_ylabel(r"$x_2$")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False)
    fig.savefig(output / "groups.png", dpi=200)
    plt.close(fig)


def run(directory, output, steps):
    torch.set_num_threads(1)
    x, observed, is_test = observed_pair_arrays(directory)
    train, test = observed[~is_test], observed[is_test]

    same_rho = minimize_scalar(lambda rho: portfolio_losses(train, rho).mean(),
                               bounds=(0, 0.95), method="bounded").x
    candidates = []
    for seed in (17, 29, 43, 61):
        q, w = train_model(x, train, steps, seed)
        loss = portfolio_losses(train, pair_correlations(q, w)).mean()
        candidates.append((loss, seed, q, w))
    _, selected_seed, q, w = min(candidates, key=lambda result: result[0])

    truth = np.genfromtxt(directory / "group_truth.csv", delimiter=",", names=True,
                         dtype=int)
    truth.sort(order="building_id")
    truth = truth["true_group"] - 1
    q, w = align_groups(q, w, truth)
    oracle = pair_correlations(np.eye(2)[truth], TRUE_WEIGHTS)
    estimates = {"independence": 0.0, "homogeneous": float(same_rho),
                 "mlp": pair_correlations(q, w), "oracle": oracle}
    loss = {model: portfolio_losses(test, rho) for model, rho in estimates.items()}
    delta = loss["homogeneous"] - loss["mlp"]
    rng = np.random.default_rng(87)
    bootstrap = delta[rng.integers(len(test), size=(5000, len(test)))].mean(axis=1)
    metrics = {
        "training_portfolios": len(train), "test_portfolios": len(test),
        "pairs_per_portfolio": len(PAIR_I), "training_steps_per_start": steps,
        "selected_seed": selected_seed,
        "training_nll_by_seed": {seed: val for val, seed, _, _ in candidates},
        "test_pair_nll": {key: float(val.mean()) for key, val in loss.items()},
        "mlp_gain_vs_homogeneous": float(delta.mean()),
        "mlp_gain_vs_homogeneous_bootstrap_95": np.quantile(bootstrap, [.025, .975]).tolist(),
        "homogeneous_rho": float(same_rho), "group_accuracy": float((q.argmax(axis=1) == truth).mean()),
        "correct_groups": int((q.argmax(axis=1) == truth).sum()),
        "true_weights": TRUE_WEIGHTS.tolist(), "fitted_weights": w.tolist(),
        "pair_rho_mae": float(np.abs(estimates["mlp"] - oracle).mean()),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "capacity_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savetxt(output / "memberships.csv", np.column_stack((np.arange(len(q)), q)),
               delimiter=",", header="building_id,q1,q2", comments="",
               fmt=["%d", "%.8f", "%.8f"])
    plot_groups(x, truth, q, output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--generate", action="store_true", help="Regenerate the fixed benchmark CSVs.")
    args = parser.parse_args()
    if args.generate:
        generate(args.data)
    run(args.data, args.out, args.steps)
