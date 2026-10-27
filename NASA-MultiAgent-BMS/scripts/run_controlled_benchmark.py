"""Controlled Controller Benchmark & Component Ablations (Phase D).
Compares:
  1. conventional (CC-CV baseline)
  2. independent (Deadline + Thermal, no health cap)
  3. independent_static_health_cap (Static 1.75 A rule cap)
  4. independent_adaptive_health (Full Adaptive Health Agent)
  5. single_agent (Tuned monolithic policy)
  6. adaptive_no_resistance (Ablation 1: without R0 input)
  7. adaptive_no_deadline_feedback (Ablation 2: without deadline floor feedback)
  8. adaptive_no_degradation_prediction (Ablation 3: without Arrhenius / high-SOC factors)

Evaluates on identical held-out scenarios (test_unseen, N=30, 3 seeds = 90 episodes per method).
Outputs:
  - results/benchmark_phase_d_episodes.csv
  - results/benchmark_phase_d_summary.csv
  - results/benchmark_phase_d_paired_tests.csv
  - results/benchmark_phase_d_distributions.png
  - results/benchmark_phase_d_ablations.png
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mabms.config import load_config
from mabms.environment.battery_env import make_scenarios
from mabms.evaluation.objective import add_J, ensure_scales
from mabms.evaluation.run_independent import run_episode

CONTROLLERS = [
    "conventional",
    "independent",
    "independent_static_health_cap",
    "independent_adaptive_health",
    "single_agent",
    "adaptive_no_resistance",
    "adaptive_no_deadline_feedback",
    "adaptive_no_degradation_prediction",
]

LABELS = {
    "conventional": "Conventional CC-CV",
    "independent": "Independent-Deadline",
    "independent_static_health_cap": "Static Health-Cap",
    "independent_adaptive_health": "Adaptive Health (Full)",
    "single_agent": "Single-Agent Policy",
    "adaptive_no_resistance": "Ablation: No R0 Input",
    "adaptive_no_deadline_feedback": "Ablation: No Deadline FB",
    "adaptive_no_degradation_prediction": "Ablation: No Deg. Pred.",
}

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

    # Load tuned single agent parameters
    best_path = res / "single_agent_best_params.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None

    n_scenarios = 30
    n_seeds = 3
    scenarios = make_scenarios(n_scenarios, seed + 200, "test_unseen")
    deadlines = {i: s.deadline_s for i, s in enumerate(scenarios)}
    soh0_map = {i: s.soh0 for i, s in enumerate(scenarios)}

    print(f"Executing Phase D Controlled Benchmark across {len(CONTROLLERS)} controllers...")
    print(f"Total episodes: {len(CONTROLLERS)} x {n_scenarios} scenarios x {n_seeds} seeds = {len(CONTROLLERS) * n_scenarios * n_seeds} runs.\n")

    episodes = []
    for k, sc in enumerate(scenarios):
        for j in range(n_seeds):
            ep_seed = 20200 + 1000 * j + k
            for method in CONTROLLERS:
                t0 = time.perf_counter()
                r = run_episode(cfg, sc, method, seed=ep_seed, safety_enabled=True, policy_params=best)
                latency_ms = (time.perf_counter() - t0) * 1000.0 / max(r["steps"], 1)

                r["scenario_idx"] = k
                r["noise_seed_idx"] = j
                r["soh0"] = soh0_map[k]
                r["is_aged"] = soh0_map[k] <= 0.80
                r["method"] = method
                r["method_label"] = LABELS[method]
                r["latency_ms_per_step"] = latency_ms
                episodes.append(r)

    df = pd.DataFrame(episodes)
    df = add_J(df, deadlines, cfg)

    # Save per-episode detailed log
    episodes_csv = res / "benchmark_phase_d_episodes.csv"
    df.to_csv(episodes_csv, index=False)
    print(f"Saved complete per-episode log to {episodes_csv}")

    # Summary table per controller
    summary_rows = []
    for m in CONTROLLERS:
        sub = df[df["method"] == m]
        aged = sub[sub["is_aged"]]
        summary_rows.append({
            "method": m,
            "label": LABELS[m],
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
    summary_csv = res / "benchmark_phase_d_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary metrics to {summary_csv}\n")
    print(summary_df[["label", "target_reached", "target_reached_aged", "time_to_target_penalized_s", "capacity_loss", "peak_temp_c", "composite_objective_J"]].to_string(index=False))

    # Paired statistical comparisons against:
    # 1. conventional
    # 2. independent
    # 3. independent_adaptive_health (for the ablations)
    paired_rows = []
    piv_J = df.pivot(index=["scenario_idx", "noise_seed_idx"], columns="method", values="J")
    piv_loss = df.pivot(index=["scenario_idx", "noise_seed_idx"], columns="method", values="capacity_loss")
    piv_time = df.pivot(index=["scenario_idx", "noise_seed_idx"], columns="method", values="time_to_target_penalized_s")

    # Comparisons against conventional
    for m in CONTROLLERS:
        if m == "conventional":
            continue
        res_j = paired_test(piv_J[m].to_numpy(), piv_J["conventional"].to_numpy())
        res_loss = paired_test(piv_loss[m].to_numpy(), piv_loss["conventional"].to_numpy())
        res_time = paired_test(piv_time[m].to_numpy(), piv_time["conventional"].to_numpy())
        paired_rows.append({
            "baseline": "conventional",
            "controller": m,
            "metric": "J",
            **res_j
        })
        paired_rows.append({
            "baseline": "conventional",
            "controller": m,
            "metric": "capacity_loss",
            **res_loss
        })
        paired_rows.append({
            "baseline": "conventional",
            "controller": m,
            "metric": "charging_time_s",
            **res_time
        })

    # Ablation comparisons against full adaptive
    for abl in ["adaptive_no_resistance", "adaptive_no_deadline_feedback", "adaptive_no_degradation_prediction"]:
        res_j = paired_test(piv_J[abl].to_numpy(), piv_J["independent_adaptive_health"].to_numpy())
        res_loss = paired_test(piv_loss[abl].to_numpy(), piv_loss["independent_adaptive_health"].to_numpy())
        res_time = paired_test(piv_time[abl].to_numpy(), piv_time["independent_adaptive_health"].to_numpy())
        paired_rows.append({
            "baseline": "independent_adaptive_health",
            "controller": abl,
            "metric": "J (Ablation Impact)",
            **res_j
        })
        paired_rows.append({
            "baseline": "independent_adaptive_health",
            "controller": abl,
            "metric": "capacity_loss (Ablation Impact)",
            **res_loss
        })
        paired_rows.append({
            "baseline": "independent_adaptive_health",
            "controller": abl,
            "metric": "charging_time_s (Ablation Impact)",
            **res_time
        })

    paired_df = pd.DataFrame(paired_rows)
    paired_csv = res / "benchmark_phase_d_paired_tests.csv"
    paired_df.to_csv(paired_csv, index=False)
    print(f"\nSaved paired statistical tests to {paired_csv}")

    # Plot 1: Performance Distributions (Composite J, Capacity Loss, Charging Time)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    plt.subplots_adjust(bottom=0.28, wspace=0.25)
    metrics_to_plot = [
        ("J", "Composite Objective J (Lower = Better)", axes[0]),
        ("capacity_loss", "Capacity Loss [x 1e-4] (Lower = Better)", axes[1]),
        ("time_to_target_penalized_s", "Charging Time [s] (Lower = Better)", axes[2]),
    ]
    short_labels = [
        "Conv.\nCC-CV",
        "Indep.\nDeadline",
        "Static\nHealth",
        "Adaptive\n(Full)",
        "Single\nAgent",
        "Abl. 1:\nNo R0",
        "Abl. 2:\nNo Deadl.",
        "Abl. 3:\nNo Deg.",
    ]
    colors = ["#7F8C8D", "#2980B9", "#E67E22", "#27AE60", "#8E44AD", "#D35400", "#C0392B", "#F39C12"]

    for col, title, ax in metrics_to_plot:
        data = []
        for m in CONTROLLERS:
            vals = df[df["method"] == m][col].to_numpy()
            if col == "capacity_loss":
                vals = vals * 1e4
            data.append(vals)
        bplot = ax.boxplot(data, tick_labels=short_labels, patch_artist=True, medianprops=dict(color="black", lw=1.8))
        for patch, clr in zip(bplot["boxes"], colors):
            patch.set_facecolor(clr)
            patch.set_alpha(0.75)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", labelsize=8.5)

    dist_png = res / "benchmark_phase_d_distributions.png"
    plt.savefig(dist_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved distribution boxplots to {dist_png}")

    # Plot 2: Component Ablation Impact Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    plt.subplots_adjust(bottom=0.2)

    abl_names = ["No R0 Input", "No Deadline FB", "No Deg. Pred."]
    abl_keys = ["adaptive_no_resistance", "adaptive_no_deadline_feedback", "adaptive_no_degradation_prediction"]
    full_j = summary_df.loc[summary_df.method == "independent_adaptive_health", "composite_objective_J"].values[0]
    full_loss = summary_df.loc[summary_df.method == "independent_adaptive_health", "capacity_loss"].values[0] * 1e4

    delta_j = [(summary_df.loc[summary_df.method == k, "composite_objective_J"].values[0] - full_j) for k in abl_keys]
    delta_loss = [(summary_df.loc[summary_df.method == k, "capacity_loss"].values[0] * 1e4 - full_loss) for k in abl_keys]

    b1 = ax1.bar(abl_names, delta_j, color=["#E67E22", "#C0392B", "#F39C12"], width=0.5)
    ax1.axhline(0, color="black", lw=1.2)
    ax1.set_ylabel("Change in Composite Cost ΔJ vs. Full Adaptive", fontsize=10.5)
    ax1.set_title("Ablation Cost Penalty (Positive = Performance Worsened)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for b in b1:
        yval = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2.0, yval + 0.005, f"{yval:+.4f}", ha="center", va="bottom", fontsize=9)

    b2 = ax2.bar(abl_names, delta_loss, color=["#E67E22", "#C0392B", "#F39C12"], width=0.5)
    ax2.axhline(0, color="black", lw=1.2)
    ax2.set_ylabel("Change in Capacity Loss ΔLoss [x 1e-4] vs. Full Adaptive", fontsize=10.5)
    ax2.set_title("Ablation Degradation Impact", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for b in b2:
        yval = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2.0, yval + 0.002, f"{yval:+.4f}", ha="center", va="bottom", fontsize=9)

    abl_png = res / "benchmark_phase_d_ablations.png"
    plt.savefig(abl_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ablation chart to {abl_png}")

if __name__ == "__main__":
    main()
