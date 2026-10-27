"""Comparative study of controllers (Baselines 1-3 now; coordinator slot reserved for later).
Usage: python scripts/06_compare_controllers.py [n_scenarios] [n_seeds]"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mabms.config import load_config
from mabms.environment.battery_env import make_scenarios
from mabms.evaluation.objective import add_J, ensure_scales
from mabms.evaluation.run_independent import run_benchmark
from mabms.evaluation.stats import compare_table, summarize
from mabms.evaluation.tune_single import tune_single_agent

cfg = load_config(); c = cfg["compare"]; seed = cfg["seed"]
n = int(sys.argv[1]) if len(sys.argv) > 1 else c["n_eval_scenarios"]; ns = int(sys.argv[2]) if len(sys.argv) > 2 else c["n_seeds"]
res = Path("results"); res.mkdir(exist_ok=True)
scales = ensure_scales(cfg); print("objective scales:", scales); (res / "objective_scales.json").write_text(json.dumps(scales))
best, trials = tune_single_agent(cfg, seed)
(res / "single_agent_best_params.json").write_text(json.dumps(best, indent=2)); pd.DataFrame([{**t["params"], "J_train": t["J_train"]} for t in trials]).to_csv(res / "single_agent_tuning_trials.csv", index=False)
print("tuned single-agent params (train/val only):", {k: round(v, 3) for k, v in best.items()})
METHODS = ["conventional", "single_agent", "independent", "independent_health_cap"]
METRICS = ["J", "time_to_target_penalized_s", "target_reached", "shortfall", "capacity_loss", "peak_temp_c", "temp_track_mae_c", "cooling_energy_wh",
           "unsafe_executed_steps", "unsafe_proposed_rate", "intervention_rate", "max_v_overshoot_v", "soc_est_mae"]
for regime, off in [("train", 100), ("test_unseen", 200)]:        # 'train' regime = in-distribution eval on fresh scenarios
    sc = make_scenarios(n, seed + off, regime)
    df = run_benchmark(cfg, sc, METHODS, base_seed=20000 + off, n_seeds=ns, policy_params=best)
    df = add_J(df, {i: s.deadline_s for i, s in enumerate(sc)}, cfg)
    df["time_to_target_s"] = df["time_to_target_s"].astype(float)       # NaN where target not reached (reported via target_reached)
    df.to_csv(res / f"compare_{regime}_episodes.csv", index=False)
    summ = summarize(df, METRICS); summ.to_csv(res / f"compare_{regime}_summary.csv")
    t = pd.concat([compare_table(df, METRICS, "conventional", ["single_agent", "independent", "independent_health_cap"]),
                   compare_table(df, METRICS, "single_agent", ["independent", "independent_health_cap"])], ignore_index=True)
    t.to_csv(res / f"compare_{regime}_paired_tests.csv", index=False)
    print(f"\n=== {regime}: n_scenarios={n}, seeds={ns} ===")
    print(summ[[(m, "mean") for m in ["J", "time_to_target_penalized_s", "capacity_loss", "unsafe_executed_steps", "target_reached"]]].round(4).to_string())
    print("\nPaired tests on J (a - b; negative => a better):")
    print(t[t.metric == "J"][["a", "b", "mean_diff", "ci95_lo", "ci95_hi", "cohens_dz", "wilcoxon_p", "p_holm"]].round(4).to_string(index=False))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, m in zip(ax, ["J", "capacity_loss", "time_to_target_penalized_s"]):
        data = [df[df.method == k].groupby("scenario_idx")[m].mean().dropna() for k in METHODS]
        a.boxplot(data); a.set_xticklabels([k.replace("independent", "indep").replace("_", "\n") for k in METHODS], fontsize=7); a.set_title(m)
    fig.suptitle(f"{regime} (per-scenario means)"); fig.tight_layout(); fig.savefig(res / f"compare_{regime}_boxplots.png", dpi=130); plt.close(fig)
