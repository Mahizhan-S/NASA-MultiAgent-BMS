"""Multi-Agent Coordinator Benchmark & Evaluation (Phase H).
Evaluates 3 coordinator architectures across held-out unseen scenarios:
1. MinimumRuleCoordinator: Baseline rule taking I = min(I_speed, I_health).
2. DynamicPriorityCoordinator: State-dependent priority switching based on deadline urgency and thermal status.
3. ConstrainedQPCoordinator: Analytic convex quadratic program balancing speed, health, and thermal loss under physical feasibility bounds.

Evaluates on N=30 test_unseen scenarios x 3 noise seeds = 90 episodes per coordinator (270 runs total).
Outputs:
  - results/coordinator_episodes.csv
  - results/coordinator_comparison.csv
  - results/coordinator_paired_tests.csv
  - results/coordinator_performance_comparison.png
"""
from __future__ import annotations
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mabms.agents.coordinator import (
    MinimumRuleCoordinator,
    DynamicPriorityCoordinator,
    ConstrainedQPCoordinator,
)
from mabms.config import load_config
from mabms.environment.battery_env import make_scenarios
from mabms.evaluation.objective import add_J, ensure_scales
from mabms.evaluation.run_independent import run_episode


def paired_test(a_vals: np.ndarray, b_vals: np.ndarray) -> dict:
    diff = a_vals - b_vals
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    se = std_diff / np.sqrt(len(diff)) if len(diff) > 0 else 0.0
    ci_lo = mean_diff - 1.96 * se
    ci_hi = mean_diff + 1.96 * se
    cohens_d = mean_diff / std_diff if std_diff > 1e-9 else 0.0
    try:
        w_stat, p_val = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    except Exception:
        p_val = 1.0
    return {
        "mean_diff": mean_diff,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "cohens_d": cohens_d,
        "wilcoxon_p": float(p_val)
    }


def main():
    cfg = load_config()
    seed = cfg["seed"]
    res = Path("results")
    res.mkdir(exist_ok=True)
    scales = ensure_scales(cfg)

    coordinators = [
        ("Minimum-Rule Coordinator", MinimumRuleCoordinator()),
        ("Dynamic-Priority Coordinator", DynamicPriorityCoordinator()),
        ("Constrained-QP Coordinator", ConstrainedQPCoordinator()),
    ]

    n_scenarios = 30
    n_seeds = 3
    scenarios = make_scenarios(n_scenarios, seed + 200, "test_unseen")
    deadlines = {i: s.deadline_s for i, s in enumerate(scenarios)}
    soh0_map = {i: s.soh0 for i, s in enumerate(scenarios)}

    print(f"Executing Phase H Coordinator Benchmark across {len(coordinators)} architectures...")
    print(f"Total episodes: {len(coordinators)} coordinators x {n_scenarios} scenarios x {n_seeds} seeds = {len(coordinators) * n_scenarios * n_seeds} runs.\n")

    episodes = []
    for coord_name, coord in coordinators:
        for k, sc in enumerate(scenarios):
            for j in range(n_seeds):
                ep_seed = 20200 + 1000 * j + k
                t0 = time.perf_counter()
                r = run_episode(
                    cfg,
                    sc,
                    method="independent_adaptive_health",
                    seed=ep_seed,
                    safety_enabled=True,
                    options={"coordinator": coord}
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0 / max(r["steps"], 1)

                r["coordinator"] = coord_name
                r["scenario_idx"] = k
                r["noise_seed_idx"] = j
                r["soh0"] = soh0_map[k]
                r["is_aged"] = soh0_map[k] <= 0.80
                r["latency_ms_per_step"] = latency_ms
                episodes.append(r)

    df = pd.DataFrame(episodes)
    df = add_J(df, deadlines, cfg)

    # Save detailed episode logs
    episodes_csv = res / "coordinator_episodes.csv"
    df.to_csv(episodes_csv, index=False)
    print(f"Saved episode data to {episodes_csv}")

    # Summary table
    summary_rows = []
    for coord_name, _ in coordinators:
        sub = df[df["coordinator"] == coord_name]
        aged = sub[sub["is_aged"]]
        summary_rows.append({
            "coordinator": coord_name,
            "episodes": len(sub),
            "target_reached": float(sub["target_reached"].mean()),
            "target_reached_aged": float(aged["target_reached"].mean()) if len(aged) > 0 else np.nan,
            "time_to_target_penalized_s": float(sub["time_to_target_penalized_s"].mean()),
            "time_to_target_std": float(sub["time_to_target_penalized_s"].std()),
            "capacity_loss": float(sub["capacity_loss"].mean()),
            "capacity_loss_std": float(sub["capacity_loss"].std()),
            "peak_temp_c": float(sub["peak_temp_c"].mean()),
            "cooling_energy_wh": float(sub["cooling_energy_wh"].mean()),
            "unsafe_executed_steps": float(sub["unsafe_executed_steps"].mean()),
            "intervention_rate": float(sub["intervention_rate"].mean()),
            "composite_objective_J": float(sub["J"].mean()),
            "composite_objective_std": float(sub["J"].std()),
            "latency_ms_per_step": float(sub["latency_ms_per_step"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = res / "coordinator_comparison.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary metrics to {summary_csv}\n")
    print(summary_df[["coordinator", "target_reached", "target_reached_aged", "time_to_target_penalized_s", "capacity_loss", "peak_temp_c", "composite_objective_J", "latency_ms_per_step"]].to_string(index=False))

    # Paired statistical comparisons against Minimum-Rule baseline
    piv_J = df.pivot(index=["scenario_idx", "noise_seed_idx"], columns="coordinator", values="J")
    piv_loss = df.pivot(index=["scenario_idx", "noise_seed_idx"], columns="coordinator", values="capacity_loss")
    piv_time = df.pivot(index=["scenario_idx", "noise_seed_idx"], columns="coordinator", values="time_to_target_penalized_s")

    paired_rows = []
    for cand in ["Dynamic-Priority Coordinator", "Constrained-QP Coordinator"]:
        res_j = paired_test(piv_J[cand].to_numpy(), piv_J["Minimum-Rule Coordinator"].to_numpy())
        res_loss = paired_test(piv_loss[cand].to_numpy(), piv_loss["Minimum-Rule Coordinator"].to_numpy())
        res_time = paired_test(piv_time[cand].to_numpy(), piv_time["Minimum-Rule Coordinator"].to_numpy())
        paired_rows.append({"baseline": "Minimum-Rule", "candidate": cand, "metric": "J", **res_j})
        paired_rows.append({"baseline": "Minimum-Rule", "candidate": cand, "metric": "capacity_loss", **res_loss})
        paired_rows.append({"baseline": "Minimum-Rule", "candidate": cand, "metric": "charging_time_s", **res_time})

    paired_df = pd.DataFrame(paired_rows)
    paired_csv = res / "coordinator_paired_tests.csv"
    paired_df.to_csv(paired_csv, index=False)
    print(f"\nSaved paired statistical tests to {paired_csv}")
    print(paired_df[["candidate", "metric", "mean_diff", "cohens_d", "wilcoxon_p"]].to_string(index=False))

    # Plot comparative distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25, wspace=0.25)

    metrics = [
        ("J", "Composite Cost J (Lower = Better)", axes[0]),
        ("capacity_loss", "Capacity Loss [x 1e-4] (Lower = Better)", axes[1]),
        ("time_to_target_penalized_s", "Charging Time [s] (Lower = Better)", axes[2]),
    ]
    coord_names = [c[0] for c in coordinators]
    colors = ["#2980B9", "#27AE60", "#8E44AD"]

    for col, title, ax in metrics:
        data = []
        for cname in coord_names:
            vals = df[df["coordinator"] == cname][col].to_numpy()
            if col == "capacity_loss":
                vals = vals * 1e4
            data.append(vals)
        bplot = ax.boxplot(data, tick_labels=[c.replace(" Coordinator", "") for c in coord_names], 
                           patch_artist=True, medianprops=dict(color="black", lw=1.8))
        for patch, clr in zip(bplot["boxes"], colors):
            patch.set_facecolor(clr)
            patch.set_alpha(0.75)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    chart_png = res / "coordinator_performance_comparison.png"
    plt.savefig(chart_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved coordinator performance chart to {chart_png}")


if __name__ == "__main__":
    main()
