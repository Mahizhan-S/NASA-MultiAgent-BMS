"""Does the model-based temperature plausibility monitor close the shared-sensor hole? Safety ON in all rows; monitor OFF vs ON; faults none / T bias -5 C /
T stuck; stress regime. Needs the v5 harness (mabms/evaluation/stress.py). Simulation only.
Usage: MABMS_CONFIG=configs/calibrated.yaml python scripts/20_temp_monitor_eval.py [n_scenarios]"""
import copy, json, sys
from pathlib import Path
import pandas as pd
from mabms.config import load_config
from mabms.environment.battery_env import make_scenarios
from mabms.evaluation.stress import make_stress_scenarios, run_stress_episode

cfg = load_config(); res = Path("results"); res.mkdir(exist_ok=True); n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
sp_path = Path("results/single_agent_best_params.json"); sp = json.loads(sp_path.read_text()) if sp_path.exists() else None
props = ["aggressive_3.5C", "independent_deadline"] + (["single_agent_tuned"] if sp else []); faults = ["none", "t_bias_-5C", "t_stuck"]
rows = []
for regime, scs in (("nominal", make_scenarios(n, cfg["seed"] + 500, "test_unseen")), ("stress", make_stress_scenarios(n, cfg["seed"] + 600))):
    for mon in (False, True):
        c = copy.deepcopy(cfg); c.setdefault("safety", {})["temp_monitor"] = {"enabled": mon, "thr_c": 4.0, "k": 3, "gain": 0.01}
        for k, sc in enumerate(scs):
            for p in props:
                for f in faults:
                    r = run_stress_episode(c, sc, p, f, True, 95000 + k, sp); r.update(regime=regime, monitor="on" if mon else "off", scenario_idx=k); rows.append(r)
df = pd.DataFrame(rows); df.to_csv(res / "v6_monitor_episodes.csv", index=False)
df["monitor_flags"] = df.get("monitor_flags", 0)
t = df.groupby(["regime", "proposer", "fault", "monitor"]).agg(unsafe_exec=("unsafe_executed_steps", "mean"), hard_viol=("hard_violation", "mean"), peak_T=("peak_temp_c", "mean"),
        t_over_tmax_s=("time_over_tmax_s", "mean"), success=("target_reached", "mean"), interventions=("intervention_rate", "mean")).round(3)
t.to_csv(res / "v6_monitor_summary.csv"); print(t.to_string())
