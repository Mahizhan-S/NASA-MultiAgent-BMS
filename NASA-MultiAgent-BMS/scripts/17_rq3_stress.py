"""RQ3: does the deterministic safety supervisor reduce unsafe-action EXECUTION? Stress regime vs nominal regime, 6 sensor-fault conditions,
3-4 proposers, safety ON vs OFF on identical scenarios/seeds (simulation only).
Usage: MABMS_CONFIG=configs/calibrated.yaml python scripts/17_rq3_stress.py [n_scenarios]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from mabms.config import load_config
from mabms.environment.battery_env import make_scenarios
from mabms.evaluation.stats import holm, paired_compare
from mabms.evaluation.stress import FAULTS, PROPOSERS, make_stress_scenarios, run_stress_grid

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=20)
ap.add_argument("--config", default=None, help="Path to YAML config file")
args, _ = ap.parse_known_args()
cfg = load_config(args.config) if args.config else load_config()
res = Path("results"); res.mkdir(exist_ok=True); n = args.n
sp_path = Path("results/single_agent_best_params.json"); sp = json.loads(sp_path.read_text()) if sp_path.exists() else None
props = PROPOSERS + (["single_agent_tuned"] if sp else [])
frames = []
for regime, scs in (("nominal", make_scenarios(n, cfg["seed"] + 500, "test_unseen")), ("stress", make_stress_scenarios(n, cfg["seed"] + 600))):
    frames.append(run_stress_grid(cfg, scs, props, FAULTS, base_seed=80000, single_params=sp, regime=regime))
df = pd.concat(frames, ignore_index=True); df.to_csv(res / "rq3_stress_episodes.csv", index=False)
agg = df.groupby(["regime", "proposer", "fault", "safety"]).agg(
    unsafe_prop_true_rate=("unsafe_prop_true_rate", "mean"), intervention_rate=("intervention_rate", "mean"), fallbacks=("fallbacks", "mean"),
    unsafe_exec_steps=("unsafe_executed_steps", "mean"), hard_viol_rate=("hard_violation", "mean"), peak_T=("peak_temp_c", "mean"),
    max_v_over=("max_v_overshoot_v", "max"), t_over_tmax_s=("time_over_tmax_s", "mean"), success=("target_reached", "mean")).round(4)
agg.to_csv(res / "rq3_stress_summary.csv")
for regime in ("nominal", "stress"):
    print(f"\n=== {regime} regime, fault=none (mean per episode) ===")
    print(agg.loc[regime].xs("none", level="fault").to_string())
rows = []
for (regime, p, f), g in df.groupby(["regime", "proposer", "fault"]):
    for metric in ("unsafe_executed_steps", "hard_violation", "peak_temp_c", "target_reached"):
        d = g.assign(method=g["safety"]); d["hard_violation"] = d["hard_violation"].astype(float); d["target_reached"] = d["target_reached"].astype(float)
        r = paired_compare(d, metric, "on", "off"); r.update(regime=regime, proposer=p, fault=f); rows.append(r)
t = pd.DataFrame(rows); t["p_holm"] = holm(t["wilcoxon_p"].fillna(1.0).tolist()); t.to_csv(res / "rq3_paired_on_vs_off.csv", index=False)
x = t[(t.metric == "unsafe_executed_steps") & (t.regime == "stress")][["proposer", "fault", "mean_a", "mean_b", "mean_diff", "ci95_lo", "ci95_hi", "p_holm"]]
print("\nStress regime: executed violation steps, safety ON (a) vs OFF (b):\n", x.round(3).to_string(index=False))
print("\nHard-limit (v_hard/t_hard) episode rate, stress regime, ON vs OFF:")
print(agg.loc["stress"].reset_index().pivot_table(index=["proposer", "fault"], columns="safety", values="hard_viol_rate").round(3).to_string())
