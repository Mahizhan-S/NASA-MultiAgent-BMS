"""Conventional vs independent specialised agents (no coordinator). Usage: python scripts/05_run_independent.py [n_scenarios]"""
import sys
from pathlib import Path
import pandas as pd
from mabms.config import load_config
from mabms.environment.battery_env import make_scenarios
from mabms.evaluation.run_independent import METHODS, run_benchmark

cfg = load_config(); n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
Path("results").mkdir(exist_ok=True)
for regime in ["train", "test_unseen"]:
    df = run_benchmark(cfg, make_scenarios(n, cfg["seed"], regime), METHODS, base_seed=1000)
    df.to_csv(f"results/independent_{regime}.csv", index=False)
    cols = ["time_to_target_s", "target_reached", "capacity_loss", "peak_temp_c", "cooling_energy_wh",
            "unsafe_executed_steps", "unsafe_proposed_rate", "intervention_rate", "soc_est_mae"]
    print(f"\n=== regime: {regime} (n={n}) mean ± std ===")
    print(df.groupby("method")[cols].agg(["mean", "std"]).round(4).T.to_string())
