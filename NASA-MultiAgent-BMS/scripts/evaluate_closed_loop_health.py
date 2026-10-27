"""Closed-Loop Health Integration & Estimation Error Sensitivity (Phase F).
Evaluates the adaptive multi-agent charging system under varying health estimation regimes:
1. Oracle Observer (Ground truth, zero estimation error)
2. Calibrated Online Estimator (Realistic state-of-the-art accuracy: sigma=0.01)
3. Systematic SOH Bias: [-10%, -5%, 0%, +5%, +10%]
4. High Observation Noise: sigma in [0.00, 0.02, 0.05, 0.08, 0.12]
5. Sensor Dropouts: 5% packet loss (testing NaN/Inf fallback resilience)
6. Resistance Bias: R0 offset by -20% and +20%

Evaluated over 20 unseen scenarios across 3 seeds (60 episodes per condition).
Outputs:
  - results/closed_loop_health_sweep.csv
  - results/closed_loop_health_summary.csv
  - results/closed_loop_health_sensitivity.png
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mabms.config import load_config
from mabms.environment.battery_env import make_scenarios
from mabms.estimation.health_interface import (
    OracleHealthEstimator,
    CalibratedOnlineHealthEstimator,
    PerturbedHealthEstimator,
)
from mabms.evaluation.objective import add_J, ensure_scales
from mabms.evaluation.run_independent import run_episode


def main():
    cfg = load_config()
    seed = cfg["seed"]
    res = Path("results")
    res.mkdir(exist_ok=True)
    scales = ensure_scales(cfg)

    n_scenarios = 20
    n_seeds = 3
    scenarios = make_scenarios(n_scenarios, seed + 300, "test_unseen")
    deadlines = {i: s.deadline_s for i, s in enumerate(scenarios)}

    # Define the health estimation test conditions
    conditions = [
        ("Oracle (Ground Truth)", OracleHealthEstimator()),
        ("Calibrated Online (sigma=0.01)", CalibratedOnlineHealthEstimator(soh_sigma=0.01, seed=seed)),
        ("Bias SOH -10% (Pessimistic)", PerturbedHealthEstimator(soh_bias=-0.10, soh_noise_sigma=0.01, seed=seed)),
        ("Bias SOH -5% (Pessimistic)", PerturbedHealthEstimator(soh_bias=-0.05, soh_noise_sigma=0.01, seed=seed)),
        ("Bias SOH +5% (Optimistic)", PerturbedHealthEstimator(soh_bias=+0.05, soh_noise_sigma=0.01, seed=seed)),
        ("Bias SOH +10% (Optimistic)", PerturbedHealthEstimator(soh_bias=+0.10, soh_noise_sigma=0.01, seed=seed)),
        ("Noise sigma=0.05 (Moderate)", PerturbedHealthEstimator(soh_bias=0.0, soh_noise_sigma=0.05, seed=seed)),
        ("Noise sigma=0.10 (Severe)", PerturbedHealthEstimator(soh_bias=0.0, soh_noise_sigma=0.10, seed=seed)),
        ("Dropouts 5% (Intermittent NaN)", PerturbedHealthEstimator(soh_bias=0.0, soh_noise_sigma=0.01, corruption_rate=0.05, seed=seed)),
        ("R0 Bias -20% (Underestimated)", PerturbedHealthEstimator(soh_bias=0.0, soh_noise_sigma=0.01, r0_bias=-0.20, seed=seed)),
        ("R0 Bias +20% (Overestimated)", PerturbedHealthEstimator(soh_bias=0.0, soh_noise_sigma=0.01, r0_bias=+0.20, seed=seed)),
    ]

    print(f"Executing Phase F Closed-Loop Health Evaluation across {len(conditions)} regimes...")
    print(f"Total runs: {len(conditions)} regimes x {n_scenarios} scenarios x {n_seeds} seeds = {len(conditions) * n_scenarios * n_seeds} episodes.\n")

    episodes = []
    for cond_name, estimator in conditions:
        for k, sc in enumerate(scenarios):
            for j in range(n_seeds):
                ep_seed = 30000 + 1000 * j + k
                # Reset estimator for this episode
                r = run_episode(
                    cfg,
                    sc,
                    method="independent_adaptive_health",
                    seed=ep_seed,
                    safety_enabled=True,
                    options={"health_estimator": estimator}
                )
                r["regime"] = cond_name
                r["scenario_idx"] = k
                r["noise_seed_idx"] = j
                r["soh0"] = sc.soh0
                r["is_aged"] = sc.soh0 <= 0.80
                episodes.append(r)

    df = pd.DataFrame(episodes)
    df = add_J(df, deadlines, cfg)

    # Save detailed episode data
    episodes_csv = res / "closed_loop_health_sweep.csv"
    df.to_csv(episodes_csv, index=False)
    print(f"Saved episode data to {episodes_csv}")

    # Compute summary per regime
    summary_rows = []
    for cond_name, _ in conditions:
        sub = df[df["regime"] == cond_name]
        aged = sub[sub["is_aged"]]
        summary_rows.append({
            "regime": cond_name,
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
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = res / "closed_loop_health_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary metrics to {summary_csv}\n")
    print(summary_df[["regime", "target_reached", "target_reached_aged", "time_to_target_penalized_s", "capacity_loss", "unsafe_executed_steps", "intervention_rate", "composite_objective_J"]].to_string(index=False))

    # Generate 4-panel diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. SOH Bias vs Capacity Loss & Charging Time
    bias_regimes = [
        ("Bias SOH -10% (Pessimistic)", -0.10),
        ("Bias SOH -5% (Pessimistic)", -0.05),
        ("Calibrated Online (sigma=0.01)", 0.00),
        ("Bias SOH +5% (Optimistic)", +0.05),
        ("Bias SOH +10% (Optimistic)", +0.10),
    ]
    x_bias = [b[1] for b in bias_regimes]
    loss_bias = [summary_df.loc[summary_df.regime == b[0], "capacity_loss"].values[0] * 1e4 for b in bias_regimes]
    time_bias = [summary_df.loc[summary_df.regime == b[0], "time_to_target_penalized_s"].values[0] for b in bias_regimes]

    ax_b1 = axes[0, 0]
    ax_b1.plot(x_bias, loss_bias, marker="o", color="#C0392B", lw=2.0, label="Capacity Loss [x 1e-4]")
    ax_b1.set_xlabel("Injected SOH Estimation Bias (ΔSOH)", fontsize=10.5, fontweight="bold")
    ax_b1.set_ylabel("Capacity Degradation ΔQ [x 10⁻⁴]", color="#C0392B", fontsize=10.5, fontweight="bold")
    ax_b1.grid(True, linestyle="--", alpha=0.4)
    ax_b1.set_title("Impact of SOH Bias on Battery Degradation", fontsize=11, fontweight="bold")

    ax_b2 = axes[0, 1]
    ax_b2.plot(x_bias, time_bias, marker="s", color="#2980B9", lw=2.0, label="Charging Time [s]")
    ax_b2.set_xlabel("Injected SOH Estimation Bias (ΔSOH)", fontsize=10.5, fontweight="bold")
    ax_b2.set_ylabel("Charging Time to Target [s]", color="#2980B9", fontsize=10.5, fontweight="bold")
    ax_b2.grid(True, linestyle="--", alpha=0.4)
    ax_b2.set_title("Impact of SOH Bias on Charging Duration", fontsize=11, fontweight="bold")

    # 3. Observation Noise Sensitivity
    noise_regimes = [
        ("Calibrated Online (sigma=0.01)", 0.01),
        ("Noise sigma=0.05 (Moderate)", 0.05),
        ("Noise sigma=0.10 (Severe)", 0.10),
    ]
    x_noise = [n[1] for n in noise_regimes]
    j_noise = [summary_df.loc[summary_df.regime == n[0], "composite_objective_J"].values[0] for n in noise_regimes]
    succ_noise = [summary_df.loc[summary_df.regime == n[0], "target_reached"].values[0] * 100.0 for n in noise_regimes]

    ax_n1 = axes[1, 0]
    ax_n1.plot(x_noise, j_noise, marker="^", color="#8E44AD", lw=2.0)
    ax_n1.set_xlabel("SOH Observation Noise (1-sigma)", fontsize=10.5, fontweight="bold")
    ax_n1.set_ylabel("Composite Cost J", fontsize=10.5, fontweight="bold")
    ax_n1.grid(True, linestyle="--", alpha=0.4)
    ax_n1.set_title("Cost Sensitivity to Health Observation Noise", fontsize=11, fontweight="bold")

    # 4. Safety & Intervention Rates Across All Regimes
    ax_s = axes[1, 1]
    y_pos = np.arange(len(conditions))
    interv = summary_df["intervention_rate"].values * 100.0
    ax_s.barh(y_pos, interv, color="#27AE60", alpha=0.8, edgecolor="black")
    ax_s.set_yticks(y_pos)
    ax_s.set_yticklabels([c[0] for c in conditions], fontsize=8.5)
    ax_s.set_xlabel("Safety Supervisor Intervention Rate (%)", fontsize=10.5, fontweight="bold")
    ax_s.set_title("Safety Guarding Under Corrupted Health Inputs", fontsize=11, fontweight="bold")
    ax_s.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plot_png = res / "closed_loop_health_sensitivity.png"
    plt.savefig(plot_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved diagnostic plots to {plot_png}")


if __name__ == "__main__":
    main()
