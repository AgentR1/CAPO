#!/usr/bin/env python3
"""Analyze local GSPO ratio diagnostics for the StepPO length-scaling pre-experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def fit_beta(rows, *, min_length: float = 0.0):
    valid = [row for row in rows if row[2] >= 30 and row[4] > 0 and row[3] >= min_length]
    if len(valid) < 2:
        return None
    x = np.log([row[3] for row in valid])
    y = np.log([row[4] for row in valid])
    beta, intercept = np.polyfit(x, y, deg=1)
    return float(beta), float(intercept), valid


def main() -> None:
    args = parse_args()
    shards = sorted(args.diagnostics_dir.glob("rank=*_pid=*_chunk=*.npz"))
    if not shards:
        raise SystemExit(f"No diagnostic shards found in {args.diagnostics_dir}")

    records = []
    for shard in shards:
        rank = int(shard.name.split("_")[0].split("=")[1])
        data = np.load(shard)
        records.append(
            np.rec.fromarrays(
                [
                    np.full(len(data["length"]), rank),
                    data["call_index"],
                    data["length"],
                    data["log_ratio_sum"],
                ],
                names="rank,call_index,length,log_ratio_sum",
            )
        )
    data = np.concatenate(records)

    # With PPO's first epoch the current policy equals the old policy before the
    # first optimizer step, so those forwards are identically (or numerically)
    # zero and contain no information about the length scaling law.
    informative_calls = []
    for call_index in np.unique(data["call_index"]):
        idx = data["call_index"] == call_index
        if np.max(np.abs(data["log_ratio_sum"][idx])) > 1e-7:
            informative_calls.append(call_index)
    data = data[np.isin(data["call_index"], informative_calls)]
    if len(data) == 0:
        raise SystemExit("All ratio sums are zero. Run with actor.ppo_epochs >= 2.")

    # Calls are synchronized across FSDP ranks.  Center within each call to
    # remove policy-drift variation before evaluating Var(S | L).
    centered = np.empty(len(data), dtype=np.float64)
    for call_index in np.unique(data["call_index"]):
        idx = data["call_index"] == call_index
        mu = data["log_ratio_sum"][idx].sum() / data["length"][idx].sum()
        centered[idx] = data["log_ratio_sum"][idx] - data["length"][idx] * mu

    edges = np.array([1, 9, 17, 33, 65, 129, 257, np.inf])
    rows = []
    bucket_values = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (data["length"] >= lo) & (data["length"] < hi)
        if idx.sum() < 2:
            continue
        variance = centered[idx].var(ddof=1)
        rows.append(
            (int(lo), "inf" if np.isinf(hi) else int(hi - 1), int(idx.sum()), data["length"][idx].mean(), variance)
        )
        bucket_values.append(centered[idx])

    print("length_bin\tn\tmean_length\tvar_centered_S\tvar_over_mean_length")
    for lo, hi, n, mean_length, variance in rows:
        print(f"[{lo},{hi}]\t{n}\t{mean_length:.3f}\t{variance:.8g}\t{variance / mean_length:.8g}")

    all_fit = fit_beta(rows)
    if all_fit is None:
        raise SystemExit("Need at least two populated length buckets to fit beta.")
    beta, intercept, all_valid = all_fit

    # L<17 consists of a tiny, behaviorally distinct set in this ALFWorld run.
    # Report a separate fit over the main supported range instead of allowing
    # those outliers to dominate the headline slope.
    main_fit = fit_beta(rows, min_length=17)
    main_ci = None
    if main_fit is not None:
        main_beta, main_intercept, main_valid = main_fit
        rng = np.random.default_rng(0)
        boot_betas = []
        selected = [(row, values) for row, values in zip(rows, bucket_values) if row in main_valid]
        for _ in range(500):
            boot_rows = []
            for row, values in selected:
                sampled = rng.choice(values, size=len(values), replace=True)
                boot_rows.append((*row[:4], sampled.var(ddof=1)))
            fit = fit_beta(boot_rows, min_length=17)
            if fit is not None:
                boot_betas.append(fit[0])
        if boot_betas:
            main_ci = np.percentile(boot_betas, [2.5, 97.5])

    print(
        f"\nbeta={beta:.4f}\nintercept={intercept:.4f}\n"
        f"shards={len(shards)}\ninformative_forwards={len(informative_calls)}\nsteps={len(data)}"
    )
    if main_fit is not None:
        ci_text = "" if main_ci is None else f" [{main_ci[0]:.4f}, {main_ci[1]:.4f}]"
        print(f"beta_main_L_ge_17={main_beta:.4f}\nbeta_main_bootstrap_95ci={ci_text.strip()}")

    output_dir = args.output_dir or args.diagnostics_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "length_variance_bins.tsv"
    with table_path.open("w") as f:
        f.write("length_bin\tn\tmean_length\tvar_centered_S\tvar_over_mean_length\n")
        for lo, hi, n, mean_length, variance in rows:
            f.write(f"[{lo},{hi}]\t{n}\t{mean_length:.6f}\t{variance:.10g}\t{variance / mean_length:.10g}\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = [f"{row[0]}–{row[1]}" for row in rows]
    counts = np.array([row[2] for row in rows])
    axes[0].bar(labels, counts, color="#4C78A8")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Step action-token length bin")
    axes[0].set_ylabel("Number of steps (log scale)")
    axes[0].set_title("ALFWorld length support")
    axes[0].tick_params(axis="x", rotation=35)

    means = np.array([row[3] for row in rows])
    variances = np.array([row[4] for row in rows])
    sizes = 35 + 125 * np.sqrt(counts / counts.max())
    axes[1].scatter(means, variances, s=sizes, c=np.log10(counts), cmap="viridis", edgecolor="black", linewidth=0.5)
    for row in rows:
        axes[1].annotate(f"n={row[2]}", (row[3], row[4]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    all_x = np.array([row[3] for row in all_valid])
    xline = np.geomspace(all_x.min(), all_x.max(), 200)
    axes[1].plot(xline, np.exp(intercept) * xline**beta, "--", color="#E45756", label=f"all eligible bins: β={beta:.2f}")
    if main_fit is not None:
        main_x = np.array([row[3] for row in main_valid])
        main_xline = np.geomspace(main_x.min(), main_x.max(), 200)
        axes[1].plot(
            main_xline,
            np.exp(main_intercept) * main_xline**main_beta,
            color="#4C78A8",
            label=f"main L≥17 bins: β={main_beta:.2f}",
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Mean action-token length in bin (log scale)")
    axes[1].set_ylabel(r"Var($S_t-L_t\mu$) (log scale)")
    axes[1].set_title("Conditional log-ratio-sum variance")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    figure_path = output_dir / "length_variance_scaling.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"table={table_path}\nfigure={figure_path}")


if __name__ == "__main__":
    main()
