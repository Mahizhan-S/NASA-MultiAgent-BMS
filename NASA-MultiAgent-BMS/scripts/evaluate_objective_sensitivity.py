"""Objective Formulation & Trade-Off Sensitivity Analysis (Phase E).
Performs:
1. 10,000-sample Dirichlet weight Monte Carlo sensitivity analysis across:
   (w_time, w_deg, w_energy) ~ Dir(1, 1, 1) and 4-way Dir(1, 1, 1, 1) including thermal stress.
2. Win-rate and preference dominance share calculation across controllers:
   - Conventional CC-CV
   - Independent-Deadline
   - Static Health-Cap (1.75 A)
   - Adaptive Health (Full Proposed)
   - Single-Agent Policy
3. Pareto dominance analysis (pairwise dominance matrix & non-dominated front identification).
4. Cross-objective trade-off correlation matrix (Spearman & Pearson rank correlation).
5. Ternary simplex visualization and Pareto frontier projection plots.

Outputs:
  - results/objective_sensitivity.csv
  - results/pareto_dominance_matrix.csv
  - results/objective_correlation_matrix.csv
  - results/dirichlet_sensitivity.png
  - results/pareto_frontier.png
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
from mabms.evaluation.objective import ensure_scales


def compute_pareto_dominance(df_means: pd.DataFrame, obj_cols: list[str]) -> pd.DataFrame:
    """Computes pairwise Pareto dominance matrix.
    Entry (A, B) is True if controller A Pareto-dominates controller B (lower is better for all).
    """
    controllers = df_means.index.tolist()
    n = len(controllers)
    dom_matrix = pd.DataFrame(False, index=controllers, columns=controllers)

    for c1 in controllers:
        v1 = df_means.loc[c1, obj_cols].values
        for c2 in controllers:
            if c1 == c2:
                continue
            v2 = df_means.loc[c2, obj_cols].values
            # c1 dominates c2 if all objectives are <= and at least one is strictly <
            if np.all(v1 <= v2) and np.any(v1 < v2):
                dom_matrix.loc[c1, c2] = True

    return dom_matrix


def project_ternary(w1: np.ndarray, w2: np.ndarray, w3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projects 3D simplex coordinates (w1 + w2 + w3 = 1) onto 2D equilateral triangle."""
    # Vertex 1 (w1=1, w2=0, w3=0) -> (0, 0)
    # Vertex 2 (w1=0, w2=1, w3=0) -> (1, 0)
    # Vertex 3 (w1=0, w2=0, w3=1) -> (0.5, sqrt(3)/2)
    x = 0.5 * (2 * w2 + w3) / (w1 + w2 + w3)
    y = (np.sqrt(3) / 2) * w3 / (w1 + w2 + w3)
    return x, y


def main():
    cfg = load_config()
    res = Path("results")
    res.mkdir(exist_ok=True)
    scales = ensure_scales(cfg)
    d_scale = scales["d_scale"]
    e_scale = scales["e_scale"]

    # Load Phase D episode records
    episodes_file = res / "benchmark_phase_d_episodes.csv"
    if not episodes_file.exists():
        raise FileNotFoundError(f"Missing {episodes_file}. Run Phase D benchmark first.")
    
    df = pd.read_csv(episodes_file)
    
    # Filter to the 5 primary comparative controllers
    primary_methods = [
        "conventional",
        "independent",
        "independent_static_health_cap",
        "independent_adaptive_health",
        "single_agent"
    ]
    labels = {
        "conventional": "Conventional CC-CV",
        "independent": "Independent-Deadline",
        "independent_static_health_cap": "Static Health-Cap",
        "independent_adaptive_health": "Adaptive Health (Full)",
        "single_agent": "Single-Agent Policy"
    }
    colors = {
        "conventional": "#7F8C8D",
        "independent": "#2980B9",
        "independent_static_health_cap": "#E67E22",
        "independent_adaptive_health": "#27AE60",
        "single_agent": "#8E44AD"
    }

    sub_df = df[df["method"].isin(primary_methods)].copy()
    
    # Compute normalized objectives per episode:
    # 1. T_norm = time_to_target_penalized_s / deadline_s
    # 2. D_norm = capacity_loss / d_scale
    # 3. E_norm = cooling_energy_wh / e_scale
    # 4. Temp_norm = max(0, peak_temp_c - 25.0) / 10.0 (thermal rise metric)
    sub_df["T_norm"] = sub_df["time_to_target_penalized_s"] / sub_df["deadline_s"]
    sub_df["D_norm"] = sub_df["capacity_loss"] / d_scale
    sub_df["E_norm"] = sub_df["cooling_energy_wh"] / e_scale
    sub_df["Temp_norm"] = np.maximum(0.0, sub_df["peak_temp_c"] - 25.0) / 10.0

    # Calculate average normalized objective vectors per controller
    means = sub_df.groupby("method")[["T_norm", "D_norm", "E_norm", "Temp_norm", 
                                      "time_to_target_penalized_s", "capacity_loss", 
                                      "cooling_energy_wh", "peak_temp_c"]].mean()
    means = means.loc[primary_methods]
    means["label"] = [labels[m] for m in means.index]

    print("--- Per-Controller Mean Normalized Objectives ---")
    print(means[["label", "T_norm", "D_norm", "E_norm", "Temp_norm"]].to_string())

    # 1. Pareto Dominance Analysis
    dom_3obj = compute_pareto_dominance(means, ["T_norm", "D_norm", "E_norm"])
    dom_4obj = compute_pareto_dominance(means, ["T_norm", "D_norm", "E_norm", "Temp_norm"])
    dom_3obj.to_csv(res / "pareto_dominance_matrix_3obj.csv")
    dom_4obj.to_csv(res / "pareto_dominance_matrix_4obj.csv")
    print("\n--- 3-Objective Pareto Dominance Matrix (Row dominates Column) ---")
    print(dom_3obj)

    # 2. Cross-Objective Correlation Analysis
    corr_spearman = sub_df[["time_to_target_penalized_s", "capacity_loss", "cooling_energy_wh", "peak_temp_c"]].corr(method="spearman")
    corr_pearson = sub_df[["time_to_target_penalized_s", "capacity_loss", "cooling_energy_wh", "peak_temp_c"]].corr(method="pearson")
    corr_spearman.to_csv(res / "objective_correlation_spearman.csv")
    corr_pearson.to_csv(res / "objective_correlation_pearson.csv")
    print("\n--- Objective Correlation Matrix (Spearman rho) ---")
    print(corr_spearman.round(3))

    # 3. Dirichlet Monte Carlo Sampling (N = 10,000)
    np.random.seed(cfg["seed"] + 777)
    N_SAMPLES = 10000
    alpha_3d = np.ones(3)
    weights_3d = np.random.dirichlet(alpha_3d, size=N_SAMPLES) # (N, 3): [w_time, w_deg, w_energy]

    # Pre-extract per-controller mean values
    T_vec = means["T_norm"].values  # shape (5,)
    D_vec = means["D_norm"].values
    E_vec = means["E_norm"].values

    # Cost tensor: J = w_t * T + w_d * D + w_e * E
    # weights: (N, 3), objectives: (3, 5) -> J: (N, 5)
    objs_3d = np.vstack([T_vec, D_vec, E_vec])
    J_monte = weights_3d @ objs_3d  # shape (N_SAMPLES, 5)

    winning_indices = np.argmin(J_monte, axis=1)
    win_counts = np.bincount(winning_indices, minlength=len(primary_methods))
    win_shares = win_counts / N_SAMPLES

    # 2-objective: Speed vs Degradation Dir(1, 1)
    weights_2d = np.random.dirichlet(np.ones(2), size=N_SAMPLES)
    objs_2d = np.vstack([T_vec, D_vec])
    J_monte_2d = weights_2d @ objs_2d
    win_shares_2d = np.bincount(np.argmin(J_monte_2d, axis=1), minlength=len(primary_methods)) / N_SAMPLES

    # 3-objective without energy: Speed vs Degradation vs Thermal Stress Dir(1, 1, 1)
    Temp_vec = means["Temp_norm"].values
    weights_sdt = np.random.dirichlet(np.ones(3), size=N_SAMPLES)
    objs_sdt = np.vstack([T_vec, D_vec, Temp_vec])
    J_monte_sdt = weights_sdt @ objs_sdt
    win_shares_sdt = np.bincount(np.argmin(J_monte_sdt, axis=1), minlength=len(primary_methods)) / N_SAMPLES

    # 4-objective Dirichlet sensitivity
    alpha_4d = np.ones(4)
    weights_4d = np.random.dirichlet(alpha_4d, size=N_SAMPLES)
    objs_4d = np.vstack([T_vec, D_vec, E_vec, Temp_vec])
    J_monte_4d = weights_4d @ objs_4d
    winning_indices_4d = np.argmin(J_monte_4d, axis=1)
    win_counts_4d = np.bincount(winning_indices_4d, minlength=len(primary_methods))
    win_shares_4d = win_counts_4d / N_SAMPLES

    sensitivity_df = pd.DataFrame({
        "method": primary_methods,
        "label": [labels[m] for m in primary_methods],
        "mean_time_s": means["time_to_target_penalized_s"].values,
        "mean_capacity_loss": means["capacity_loss"].values,
        "mean_cooling_energy_wh": means["cooling_energy_wh"].values,
        "mean_peak_temp_c": means["peak_temp_c"].values,
        "win_speed_vs_deg_pct": win_shares_2d * 100.0,
        "win_speed_deg_temp_pct": win_shares_sdt * 100.0,
        "win_share_3obj_pct": win_shares * 100.0,
        "win_share_4obj_pct": win_shares_4d * 100.0,
    })
    sensitivity_df.to_csv(res / "objective_sensitivity.csv", index=False)
    print("\n--- Dirichlet Weight Sensitivity Results Across Preference Regimes (10,000 samples) ---")
    print(sensitivity_df[["label", "win_speed_vs_deg_pct", "win_speed_deg_temp_pct", "win_share_3obj_pct", "win_share_4obj_pct"]].to_string(index=False))

    # 4. Generate Plot 1: Dirichlet Sensitivity & Ternary Simplex
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.3])

    # Left: Win share bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(primary_methods))
    w_bar = 0.35
    b1 = ax1.bar(x_pos - w_bar/2, sensitivity_df["win_share_3obj_pct"], width=w_bar, 
                 label="3-Obj (Time, Deg, Energy)", color="#3498DB", alpha=0.85, edgecolor="black")
    b2 = ax1.bar(x_pos + w_bar/2, sensitivity_df["win_share_4obj_pct"], width=w_bar, 
                 label="4-Obj (+ Thermal Stress)", color="#E67E22", alpha=0.85, edgecolor="black")
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([labels[m].replace(" ", "\n") for m in primary_methods], fontsize=9)
    ax1.set_ylabel("Win Share Across 10,000 Random Weights (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Controller Dominance Under Uniform Dirichlet Preferences", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9.5, loc="upper right")
    
    for bar in list(b1) + list(b2):
        yval = bar.get_height()
        if yval > 0.5:
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5)

    # Right: Ternary Simplex Winner Partition Map
    ax2 = fig.add_subplot(gs[0, 1])
    # Project sample weights to 2D
    # w_time = weights_3d[:, 0], w_energy = weights_3d[:, 1], w_deg = weights_3d[:, 2]
    w_t = weights_3d[:, 0]
    w_e = weights_3d[:, 1]
    w_d = weights_3d[:, 2]
    px, py = project_ternary(w_t, w_e, w_d)

    # Plot simplex boundary
    tri_x = [0.0, 1.0, 0.5, 0.0]
    tri_y = [0.0, 0.0, np.sqrt(3)/2, 0.0]
    ax2.plot(tri_x, tri_y, color="black", lw=1.8)

    # Subsample 4,000 points for clear rendering
    sub_idx = np.random.choice(N_SAMPLES, 4000, replace=False)
    for idx_m, m in enumerate(primary_methods):
        mask = (winning_indices[sub_idx] == idx_m)
        if np.any(mask):
            ax2.scatter(px[sub_idx][mask], py[sub_idx][mask], color=colors[m], label=labels[m], s=12, alpha=0.65)

    # Vertex labels
    ax2.text(-0.04, -0.04, "w_time = 1.0\n(Fast Charging)", ha="left", va="top", fontsize=10, fontweight="bold")
    ax2.text(1.04, -0.04, "w_energy = 1.0\n(Min. Cooling)", ha="right", va="top", fontsize=10, fontweight="bold")
    ax2.text(0.5, np.sqrt(3)/2 + 0.03, "w_deg = 1.0\n(Min. Degradation)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax2.set_xlim(-0.15, 1.15)
    ax2.set_ylim(-0.12, 1.05)
    ax2.axis("off")
    ax2.set_title("Winning Controller Partition on Ternary Simplex", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    dirichlet_png = res / "dirichlet_sensitivity.png"
    plt.savefig(dirichlet_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved Dirichlet sensitivity visualization to {dirichlet_png}")

    # 5. Generate Plot 2: Multi-Objective Pareto Frontier Projections
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Subplot 1: Charging Time vs Capacity Loss
    ax_p1 = axes[0]
    for m in primary_methods:
        row = means.loc[m]
        sub_pts = sub_df[sub_df["method"] == m]
        # Scatter individual episodes lightly
        ax_p1.scatter(sub_pts["time_to_target_penalized_s"], sub_pts["capacity_loss"] * 1e4, 
                      color=colors[m], alpha=0.18, s=18)
        # Plot mean centroid with star
        ax_p1.scatter(row["time_to_target_penalized_s"], row["capacity_loss"] * 1e4, 
                      color=colors[m], edgecolor="black", s=180, marker="*", label=labels[m], zorder=5)
    
    ax_p1.set_xlabel("Penalized Charging Duration [s] (Lower = Better)", fontsize=10.5, fontweight="bold")
    ax_p1.set_ylabel("Capacity Degradation ΔQ [x 10⁻⁴] (Lower = Better)", fontsize=10.5, fontweight="bold")
    ax_p1.set_title("Pareto Trade-Off: Speed vs. Degradation", fontsize=11.5, fontweight="bold")
    ax_p1.grid(True, linestyle="--", alpha=0.4)
    ax_p1.legend(loc="upper right", fontsize=8.5)

    # Subplot 2: Capacity Loss vs Peak Temperature
    ax_p2 = axes[1]
    for m in primary_methods:
        row = means.loc[m]
        sub_pts = sub_df[sub_df["method"] == m]
        ax_p2.scatter(sub_pts["peak_temp_c"], sub_pts["capacity_loss"] * 1e4, 
                      color=colors[m], alpha=0.18, s=18)
        ax_p2.scatter(row["peak_temp_c"], row["capacity_loss"] * 1e4, 
                      color=colors[m], edgecolor="black", s=180, marker="*", label=labels[m], zorder=5)

    ax_p2.set_xlabel("Peak Cell Temperature [°C] (Lower = Better)", fontsize=10.5, fontweight="bold")
    ax_p2.set_ylabel("Capacity Degradation ΔQ [x 10⁻⁴] (Lower = Better)", fontsize=10.5, fontweight="bold")
    ax_p2.set_title("Pareto Trade-Off: Thermal Rise vs. Degradation", fontsize=11.5, fontweight="bold")
    ax_p2.grid(True, linestyle="--", alpha=0.4)
    ax_p2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    pareto_png = res / "pareto_frontier.png"
    plt.savefig(pareto_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Pareto trade-off projection to {pareto_png}")


if __name__ == "__main__":
    main()
