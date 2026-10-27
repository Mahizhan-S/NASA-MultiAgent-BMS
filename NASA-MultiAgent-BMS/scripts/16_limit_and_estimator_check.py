"""(A) current-limit consistency: executed current-limit violations must be ~0 with safety ON once the limit is absolute.
(B) SOC-estimator ablation: fixed-capacity EKF vs capacity-adaptive EKF on IDENTICAL scenarios/seeds (test_unseen regime).
Usage: MABMS_CONFIG=configs/calibrated.yaml python scripts/16_limit_and_estimator_check.py [n_scenarios] [n_seeds]"""
import copy, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from mabms.config import load_config
from mabms.environment.battery_env import make_scenarios
from mabms.evaluation.run_independent import run_benchmark
from mabms.evaluation.stats import paired_compare
from mabms.limits import current_limit_a

cfg = load_config(); res = Path("results"); res.mkdir(exist_ok=True)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 30; ns = int(sys.argv[2]) if len(sys.argv) > 2 else 3
bp = json.loads(Path("results/single_agent_best_params.json").read_text()) if Path("results/single_agent_best_params.json").exists() else None
methods = ["conventional", "independent", "independent_health_cap"] + (["single_agent"] if bp else [])
sc = make_scenarios(n, cfg["seed"] + 200, "test_unseen"); q = np.array([s.q_scale for s in sc])
print(f"absolute current limit = {current_limit_a(cfg):.3f} A")
rows = []
for adaptive in (False, True):
    c = copy.deepcopy(cfg); c.setdefault("estimator", {})["adaptive_capacity"] = adaptive
    d = run_benchmark(c, sc, methods, base_seed=70000, n_seeds=ns, policy_params=bp); d["adaptive"] = adaptive
    d["label"] = np.where(adaptive, "adaptive_ekf", "fixed_ekf"); d["q_scale"] = q[d.scenario_idx.to_numpy()]; rows.append(d)
df = pd.concat(rows, ignore_index=True); df["abs_final_err"] = df.soc_est_final_err.abs(); df.to_csv(res / "limit_estimator_check_episodes.csv", index=False)
print("\n(A) executed current-limit violation steps (safety ON), mean per episode:")
print(df.groupby(["method", "label"])["sim_violations_i"].mean().unstack().round(3).to_string())
t = df.groupby(["method", "label"]).agg(success=("target_reached", "mean"), shortfall=("shortfall", "mean"), abs_final_soc_err=("abs_final_err", "mean"),
        final_err_bias=("soc_est_final_err", "mean"), soc_mae=("soc_est_mae", "mean"), time_pen_s=("time_to_target_penalized_s", "mean"),
        unsafe_exec=("unsafe_executed_steps", "mean")).round(4)
t.to_csv(res / "estimator_ablation_summary.csv"); print("\n(B) fixed vs adaptive EKF:\n", t.to_string())
cmp = []
for m in methods:
    sub = df[df.method == m].assign(method_=lambda x: x["label"])
    for metric in ("abs_final_err", "shortfall", "time_to_target_penalized_s"):
        r = paired_compare(sub.drop(columns="method").rename(columns={"method_": "method"}), metric, "adaptive_ekf", "fixed_ekf"); r["controller"] = m; cmp.append(r)
c = pd.DataFrame(cmp); c.to_csv(res / "estimator_ablation_paired.csv", index=False)
print("\nPaired (adaptive - fixed; negative => adaptive better):\n", c[["controller", "metric", "mean_diff", "ci95_lo", "ci95_hi", "wilcoxon_p"]].round(4).to_string(index=False))
