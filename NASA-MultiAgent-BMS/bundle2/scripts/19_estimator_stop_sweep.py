"""One-factor-at-a-time sweep of the v6 options on IDENTICAL test_unseen scenarios/seeds (simulation only). Baseline = whatever the loaded config says
(v4: capacity-adaptive EKF). Usage: MABMS_CONFIG=configs/calibrated.yaml python scripts/19_estimator_stop_sweep.py [n_scenarios] [n_seeds]"""
import copy, sys
from pathlib import Path
import numpy as np, pandas as pd
from mabms.config import load_config
from mabms.environment.battery_env import make_scenarios
from mabms.evaluation.run_independent import run_benchmark
from mabms.evaluation.stats import paired_compare

cfg = load_config(); res = Path("results"); res.mkdir(exist_ok=True)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20; ns = int(sys.argv[2]) if len(sys.argv) > 2 else 2
def variant(**kw):
    c = copy.deepcopy(cfg); c.setdefault("estimator", {}); c.setdefault("safety", {}); c["agents"].setdefault("charging", {})
    c["estimator"]["adapt_r0"] = False; c["safety"]["r0_margin"] = 1.0
    c["agents"]["charging"]["restart_hysteresis"] = 0.0; c["agents"]["charging"]["soc_stop_margin"] = 0.0
    c["estimator"].update(kw.get("est", {})); c["safety"].update(kw.get("safety", {})); c["agents"]["charging"].update(kw.get("charging", {})); return c
V = {"A_baseline": variant(est={"adaptive_capacity": True}),
     "B_adapt_R0": variant(est={"adaptive_capacity": True, "adapt_r0": True}),
     "C_stop_latch_0.02": variant(est={"adaptive_capacity": True}, charging={"restart_hysteresis": 0.02}),
     "D_latch+margin_0.01": variant(est={"adaptive_capacity": True}, charging={"restart_hysteresis": 0.02, "soc_stop_margin": 0.01}),
     "E_safety_r0_margin_1.5": variant(est={"adaptive_capacity": True}, safety={"r0_margin": 1.5}),
     "F_R0+r0margin_1.3": variant(est={"adaptive_capacity": True, "adapt_r0": True}, safety={"r0_margin": 1.3}),
     "G_R0+r0margin_1.3+stop_margin_0.005": variant(est={"adaptive_capacity": True, "adapt_r0": True}, charging={"soc_stop_margin": 0.005}, safety={"r0_margin": 1.3}),
     "H_R0+latch+r0margin_1.3": variant(est={"adaptive_capacity": True, "adapt_r0": True}, charging={"restart_hysteresis": 0.02}, safety={"r0_margin": 1.3})}
sc = make_scenarios(n, cfg["seed"] + 200, "test_unseen"); frames = []
for name, c in V.items():
    d = run_benchmark(c, sc, ["conventional", "independent"], base_seed=90000, n_seeds=ns); d["variant"] = name; frames.append(d)
df = pd.concat(frames, ignore_index=True); df["abs_final_err"] = df.soc_est_final_err.abs(); df["target_reached"] = df.target_reached.astype(float); df.to_csv(res / "v6_sweep_episodes.csv", index=False)
t = df.groupby(["method", "variant"]).agg(success=("target_reached", "mean"), shortfall=("shortfall", "mean"), abs_final_err=("abs_final_err", "mean"), err_bias=("soc_est_final_err", "mean"),
        unsafe_exec=("unsafe_executed_steps", "mean"), max_v_over=("max_v_overshoot_v", "max"), time_pen_s=("time_to_target_penalized_s", "mean")).round(4)
t.to_csv(res / "v6_sweep_summary.csv"); print(t.to_string())
rows = []
for m in ["conventional", "independent"]:
    for v in list(V)[1:]:
        sub = df[(df.method == m) & (df.variant.isin(["A_baseline", v]))].rename(columns={"method": "m", "variant": "method"})
        for metric in ("target_reached", "abs_final_err", "unsafe_executed_steps", "time_to_target_penalized_s"):
            r = paired_compare(sub, metric, v, "A_baseline"); r["controller"] = m; rows.append(r)
p = pd.DataFrame(rows); p.to_csv(res / "v6_sweep_paired_vs_baseline.csv", index=False)
print("\nPaired vs baseline A (variant - A):\n", p[["controller", "a", "metric", "mean_diff", "ci95_lo", "ci95_hi", "wilcoxon_p"]].round(4).to_string(index=False))
