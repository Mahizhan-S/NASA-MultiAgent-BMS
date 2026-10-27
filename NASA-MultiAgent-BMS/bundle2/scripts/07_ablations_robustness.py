"""Ablations of the independent-agent system + sensor-noise robustness. (Coordinator ablations come after the coordinator exists.)
Usage: python scripts/07_ablations_robustness.py [n_scenarios] [n_seeds]"""
import copy, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mabms.config import load_config
from mabms.environment.battery_env import make_scenarios
from mabms.evaluation.objective import add_J, ensure_scales
from mabms.evaluation.run_independent import run_benchmark
from mabms.evaluation.stats import compare_table, summarize

cfg = load_config(); seed = cfg["seed"]; res = Path("results"); res.mkdir(exist_ok=True)
print("objective scales:", ensure_scales(cfg))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20; ns = int(sys.argv[2]) if len(sys.argv) > 2 else 2
sc = make_scenarios(n, seed + 300, "test_unseen"); dl = {i: s.deadline_s for i, s in enumerate(sc)}
bp = json.loads(Path("results/single_agent_best_params.json").read_text()) if Path("results/single_agent_best_params.json").exists() else None
FULL = "independent_health_cap"
variants = [("full", dict(methods=[FULL])), ("no_health_agent", dict(methods=[FULL], options={"no_health": True})),
            ("no_thermal_agent", dict(methods=[FULL], options={"no_thermal": True})),
            ("safety_off(sim-only)", dict(methods=[FULL], safety_enabled=False))]
dfs = [run_benchmark(cfg, sc, base_seed=30000, n_seeds=ns, label=lab, **kw) for lab, kw in variants]
df = add_J(pd.concat(dfs, ignore_index=True), dl, cfg); df["time_to_target_s"] = df["time_to_target_s"].astype(float)
M = ["J", "time_to_target_penalized_s", "target_reached", "shortfall", "capacity_loss", "peak_temp_c", "cooling_energy_wh", "unsafe_executed_steps", "max_v_overshoot_v"]
df.to_csv(res / "ablation_episodes.csv", index=False); summarize(df, M).to_csv(res / "ablation_summary.csv")
t = compare_table(df, M, "full", [v[0] for v in variants[1:]]); t.to_csv(res / "ablation_paired_tests.csv", index=False)
print("=== ablations (test_unseen) ===\n", summarize(df, ["J", "capacity_loss", "peak_temp_c", "unsafe_executed_steps", "target_reached"]).to_string())
print("\nPaired vs full (J):\n", t[t.metric == "J"][["a", "mean_diff", "ci95_lo", "ci95_hi", "wilcoxon_p", "p_holm"]].round(4).to_string(index=False))

# ---- sensor-noise robustness: scale all measurement sigmas
rows = []
meths = ["conventional", "independent", FULL] + (["single_agent"] if bp else [])
for scale in [0.0, 1.0, 2.0, 4.0]:
    c2 = copy.deepcopy(cfg)
    for k in ("v_sigma", "i_sigma", "t_sigma", "soh_sigma"): c2["noise"][k] *= scale
    d = run_benchmark(c2, sc, meths, base_seed=40000, n_seeds=ns, policy_params=bp); d["noise_scale"] = scale; rows.append(d)
rb = add_J(pd.concat(rows, ignore_index=True), dl, cfg); rb.to_csv(res / "robustness_noise_episodes.csv", index=False)
agg = rb.groupby(["method", "noise_scale"])[["J", "unsafe_executed_steps", "soc_est_mae", "target_reached", "max_v_overshoot_v"]].mean().round(4)
agg.to_csv(res / "robustness_noise_summary.csv"); print("\n=== noise robustness ===\n", agg.to_string())
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for m in meths:
    g = agg.loc[m]; ax[0].plot(g.index, g["J"], marker="o", label=m); ax[1].plot(g.index, g["unsafe_executed_steps"], marker="o", label=m)
ax[0].set_title("J vs noise scale"); ax[1].set_title("unsafe executed steps vs noise scale"); ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(res / "robustness_noise.png", dpi=130)
