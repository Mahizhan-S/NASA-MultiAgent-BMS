"""Degradation Model Validation and Stress-Sweep Script (Phase C).
Evaluates:
  1. Parametric response under controlled sweeps (current, temperature, SOC).
  2. Prediction vs empirical reference checkpoints on NASA PCoE validation packs.
  3. Regime-specific error analysis and model validity bounds.
Outputs:
  - results/degradation_validation_curves.png
  - results/degradation_validation_metrics.json
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
from mabms.environment.battery_model import BatteryModel, ModelParams
from mabms.environment.degradation_fit import T_REF, K0

def evaluate_parametric_response(cfg):
    """Run controlled sweeps over current, temperature, and SOC to verify plausible physical response."""
    p = ModelParams.from_config(cfg)
    c_rates = np.linspace(0.2, 2.0, 50)
    temps = [15.0, 25.0, 35.0, 45.0]
    socs = [0.3, 0.6, 0.85, 0.95]

    # 1. dLoss vs C-rate across temperatures (at SOC = 0.5)
    curve_c_temp = {}
    for T in temps:
        dloss_list = []
        for c in c_rates:
            f_T = np.exp(p.ea_deg_k * (1.0 / (T_REF + K0) - 1.0 / (T + K0)))
            f_C = 1.0 + p.c_rate_coeff * max(c - 0.5, 0.0)
            f_S = 1.0 # SOC = 0.5 <= 0.8
            # 1 hour equivalent throughput
            dloss = p.k_deg * (c * p.q0_ah * 3600.0 / 3600.0 / p.q0_ah) * f_T * f_C * f_S
            dloss_list.append(dloss)
        curve_c_temp[T] = dloss_list

    # 2. dLoss vs SOC across C-rates (at T = 25C)
    soc_range = np.linspace(0.1, 1.0, 50)
    curve_soc = {}
    for c in [0.5, 1.0, 1.5]:
        dloss_list = []
        for s in soc_range:
            f_T = 1.0 # T = 25C
            f_C = 1.0 + p.c_rate_coeff * max(c - 0.5, 0.0)
            f_S = 1.0 + p.soc_stress * (max(s - 0.8, 0.0) / 0.2) ** 2
            dloss = p.k_deg * (c * p.q0_ah * 3600.0 / 3600.0 / p.q0_ah) * f_T * f_C * f_S
            dloss_list.append(dloss)
        curve_soc[c] = dloss_list

    return c_rates, curve_c_temp, soc_range, curve_soc

def main():
    cfg = load_config()
    res = Path("results")
    res.mkdir(exist_ok=True)
    p = ModelParams.from_config(cfg)

    # 1. Parametric response
    c_rates, curve_c_temp, soc_range, curve_soc = evaluate_parametric_response(cfg)

    # 2. Load empirical validation pack fits if available
    val_pack_csv = res / "degradation_val_per_pack.csv"
    val_variants_csv = res / "degradation_variants_val.csv"
    bootstrap_csv = res / "degradation_bootstrap_ci.csv"

    val_packs_df = pd.read_csv(val_pack_csv) if val_pack_csv.exists() else None
    val_variants_df = pd.read_csv(val_variants_csv) if val_variants_csv.exists() else None
    bootstrap_df = pd.read_csv(bootstrap_csv) if bootstrap_csv.exists() else None

    # Plot 4-panel validation figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    # Panel (a): Degradation rate vs C-rate across temperatures
    ax1 = axes[0, 0]
    colors = ["#2980B9", "#27AE60", "#E67E22", "#C0392B"]
    for T, col in zip(sorted(curve_c_temp.keys()), colors):
        ax1.plot(c_rates, np.array(curve_c_temp[T]) * 1e4, label=f"T = {T:.0f} °C", color=col, lw=2.2)
    ax1.set_xlabel("Charging C-Rate [C]", fontsize=10.5)
    ax1.set_ylabel("Degradation Rate [$10^{-4}$ loss / 1 Ah]", fontsize=10.5)
    ax1.set_title("(a) Arrhenius & Kinetic Degradation vs. C-Rate", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=9)

    # Panel (b): High-SOC stress acceleration
    ax2 = axes[0, 1]
    soc_colors = ["#16A085", "#2980B9", "#8E44AD"]
    for c, col in zip([0.5, 1.0, 1.5], soc_colors):
        ax2.plot(soc_range, np.array(curve_soc[c]) * 1e4, label=f"C = {c:.1f} C", color=col, lw=2.2)
    ax2.axvline(0.80, color="crimson", linestyle=":", lw=1.8, label="Plating Threshold (SOC=0.80)")
    ax2.set_xlabel("State of Charge (SOC)", fontsize=10.5)
    ax2.set_ylabel("Degradation Rate [$10^{-4}$ loss / 1 Ah]", fontsize=10.5)
    ax2.set_title("(b) Anode Stress & Overpotential vs. SOC", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(frameon=True, fontsize=9)

    # Panel (c): Model Variant Selection on NASA Validation Packs
    ax3 = axes[1, 0]
    if val_variants_df is not None:
        variants = val_variants_df["variant"].tolist()
        maes = (val_variants_df["val_mae_soh"] * 100).tolist() # in percentage SOH
        bars = ax3.bar(variants, maes, color=["#BDC3C7", "#95A5A6", "#3498DB", "#2ECC71", "#E74C3C"], width=0.55)
        ax3.set_ylabel("Validation SOH MAE [%]", fontsize=10.5)
        ax3.set_title("(c) Validation Pack Model Selection (MAE)", fontsize=11, fontweight="bold")
        ax3.set_xticklabels(variants, rotation=15, ha="right", fontsize=9)
        ax3.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in bars:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}%", ha="center", va="bottom", fontsize=8.5)

    # Panel (d): Per-Pack Error Breakdown
    ax4 = axes[1, 1]
    if val_packs_df is not None:
        packs = val_packs_df["pack_id"].tolist()
        p_maes = (val_packs_df["mae_soh"] * 100).tolist()
        p_rmse = (val_packs_df["rmse_soh"] * 100).tolist()
        x = np.arange(len(packs))
        w = 0.35
        ax4.bar(x - w/2, p_maes, width=w, label="MAE [%]", color="#3498DB")
        ax4.bar(x + w/2, p_rmse, width=w, label="RMSE [%]", color="#E67E22")
        ax4.set_xticks(x)
        ax4.set_xticklabels(packs, fontsize=9.5)
        ax4.set_ylabel("SOH Tracking Error [%]", fontsize=10.5)
        ax4.set_title("(d) Error Breakdown on Validation Packs", fontsize=11, fontweight="bold")
        ax4.grid(axis="y", linestyle="--", alpha=0.4)
        ax4.legend(frameon=True, fontsize=9)

    out_png = res / "degradation_validation_curves.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved validation figure to {out_png}")

    # Metrics dictionary
    metrics = {
        "model_form": "dLoss = k * (Ah_throughput / Q0) * f_T * f_C * f_S",
        "parameters": {
            "k_deg": p.k_deg,
            "ea_over_r_k": p.ea_deg_k,
            "apparent_activation_energy_kj_per_mol": round(p.ea_deg_k * 8.314 / 1000.0, 2),
            "c_rate_coeff": p.c_rate_coeff,
            "soc_stress": p.soc_stress,
            "t_ref_c": T_REF
        },
        "variant_performance": val_variants_df.to_dict("records") if val_variants_df is not None else {},
        "per_pack_validation": val_packs_df.to_dict("records") if val_packs_df is not None else {},
        "bootstrap_95_ci": bootstrap_df.to_dict() if bootstrap_df is not None else {},
        "model_validity_bounds": {
            "c_rate_min": 0.1,
            "c_rate_max": 2.5,
            "temp_c_min": 15.0,
            "temp_c_max": 45.0,
            "soc_min": 0.1,
            "soc_max": 0.98
        }
    }
    out_json = res / "degradation_validation_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"Saved validation metrics to {out_json}")

if __name__ == "__main__":
    main()
