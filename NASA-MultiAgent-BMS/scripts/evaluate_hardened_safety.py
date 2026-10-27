"""Safety Supervisor Hardening Evaluation (Phase G).
Benchmarks:
1. Safety Supervisor OFF (Unprotected plant baseline)
2. Safety Supervisor Unhardened (Baseline from Thesis Chapter 6, Table 6.3)
3. Safety Supervisor Hardened (Proposed multi-layer temporal & energy-balance monitor)

Evaluates under all 7 sensor fault regimes from Chapter 6:
  - none
  - t_bias_-5C (The critical failure case causing 4.2 unsafe steps in Table 6.3)
  - t_bias_onset_-5C
  - t_stuck
  - t_dropout_5pct
  - v_bias_+0.1V
  - v_dropout_5pct

Under aggressive 3.5 C unconstrained proposer and independent deadline-aware proposer.
Outputs:
  - results/fault_injection_hardened_vs_unhardened.csv
  - results/fault_injection_summary.csv
  - results/safety_hardening_comparison.png
"""
from __future__ import annotations
import copy
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mabms.config import load_config
from mabms.evaluation.stress import (
    FAULTS,
    make_stress_scenarios,
    run_stress_episode,
)


def run_evaluation(cfg: dict, scs: list, faults: list[str], modes: list[tuple[str, bool, bool]], proposer: str = "aggressive_3.5C") -> pd.DataFrame:
    rows = []
    for mode_name, safety_on, hardened in modes:
        c = copy.deepcopy(cfg)
        c.setdefault("safety", {})["hardened"] = hardened
        for k, sc in enumerate(scs):
            for seed in [101, 102, 103]:
                for f in faults:
                    res = run_stress_episode(
                        c,
                        sc,
                        proposer=proposer,
                        fault=f,
                        safety_enabled=safety_on,
                        seed=seed
                    )
                    res["mode"] = mode_name
                    res["scenario_idx"] = k
                    res["proposer"] = proposer
                    rows.append(res)
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    res = Path("results")
    res.mkdir(exist_ok=True)

    n_scenarios = 10
    scenarios = make_stress_scenarios(n_scenarios, cfg["seed"] + 600)

    modes = [
        ("Safety OFF", False, False),
        ("Unhardened Supervisor", True, False),
        ("Hardened Supervisor", True, True),
    ]

    print(f"Executing Phase G Safety Hardening Evaluation across {len(modes)} supervisor modes and {len(FAULTS)} fault regimes...")
    print(f"Total runs: {len(modes)} modes x {len(FAULTS)} faults x {n_scenarios} scenarios x 3 seeds = {len(modes) * len(FAULTS) * n_scenarios * 3} episodes.\n")

    df = run_evaluation(cfg, scenarios, FAULTS, modes, proposer="aggressive_3.5C")

    # Save detailed episode logs
    raw_csv = res / "fault_injection_hardened_vs_unhardened.csv"
    df.to_csv(raw_csv, index=False)
    print(f"Saved raw episode logs to {raw_csv}")

    # Summary table: Mean unsafe executed steps, hard violations, peak temperature
    agg = df.groupby(["mode", "fault"]).agg(
        unsafe_executed_steps=("unsafe_executed_steps", "mean"),
        hard_violation_rate=("hard_violation", "mean"),
        peak_temp_c=("peak_temp_c", "mean"),
        time_over_tmax_s=("time_over_tmax_s", "mean"),
        intervention_rate=("intervention_rate", "mean"),
        monitor_flags=("monitor_flags", "mean")
    ).reset_index()

    summary_csv = res / "fault_injection_summary.csv"
    agg.to_csv(summary_csv, index=False)
    print(f"Saved summary metrics to {summary_csv}\n")

    # Display key comparison table
    piv = agg.pivot(index="fault", columns="mode", values="unsafe_executed_steps")
    print("--- Unsafe Executed Steps Comparison (Target: 0.0) ---")
    print(piv.round(2).to_string())

    # Generate comparative bar chart
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    plt.subplots_adjust(bottom=0.25, wspace=0.25)

    x = np.arange(len(FAULTS))
    w = 0.26

    # Panel 1: Unsafe Executed Steps
    ax1 = axes[0]
    for idx, (mode_name, _, _) in enumerate(modes):
        sub = agg[agg["mode"] == mode_name].set_index("fault").loc[FAULTS]
        color = "#C0392B" if "OFF" in mode_name else ("#E67E22" if "Unhardened" in mode_name else "#27AE60")
        bars = ax1.bar(x + (idx - 1) * w, sub["unsafe_executed_steps"], width=w, label=mode_name, color=color, alpha=0.85, edgecolor="black")
        for b in bars:
            h = b.get_height()
            if h > 0.05:
                ax1.text(b.get_x() + b.get_width()/2.0, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f.replace("_", "\n") for f in FAULTS], fontsize=8.5)
    ax1.set_ylabel("Unsafe Executed Steps (Lower = Safer)", fontsize=10.5, fontweight="bold")
    ax1.set_title("Unsafe Steps Executed Under Sensor Faults", fontsize=11.5, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9, loc="upper right")

    # Panel 2: Peak Cell Temperature (°C)
    ax2 = axes[1]
    for idx, (mode_name, _, _) in enumerate(modes):
        sub = agg[agg["mode"] == mode_name].set_index("fault").loc[FAULTS]
        color = "#C0392B" if "OFF" in mode_name else ("#E67E22" if "Unhardened" in mode_name else "#27AE60")
        ax2.bar(x + (idx - 1) * w, sub["peak_temp_c"], width=w, label=mode_name, color=color, alpha=0.85, edgecolor="black")

    ax2.axhline(45.0, color="#C0392B", linestyle="--", lw=1.5, label="Max Thermal Limit (45 °C)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.replace("_", "\n") for f in FAULTS], fontsize=8.5)
    ax2.set_ylabel("Peak Cell Temperature [°C]", fontsize=10.5, fontweight="bold")
    ax2.set_title("Thermal Safety Invariant Under Sensor Faults", fontsize=11.5, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.set_ylim(35.0, 52.0)
    ax2.legend(fontsize=8.5, loc="upper left")

    plt.tight_layout()
    chart_png = res / "safety_hardening_comparison.png"
    plt.savefig(chart_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved hardening comparison chart to {chart_png}")


if __name__ == "__main__":
    main()
