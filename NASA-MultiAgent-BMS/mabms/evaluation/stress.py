"""RQ3 stress harness: limit-binding scenarios, an UNCONSTRAINED aggressive proposer, and sensor faults. Compares safety ON vs OFF
on identical scenarios/seeds. Three quantities are kept separate (as the study requires):
  * unsafe_prop_true_steps : proposals that would violate a limit w.r.t. the TRUE plant state (diagnostic access to env.model)
  * interventions / fallbacks : steps where the safety agent changed the command (based on its possibly-faulty estimate)
  * unsafe_executed_steps    : steps where the simulated plant actually violated a limit
SIMULATION ONLY. Not a certification of any real BMS."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ..agents.base import StateEstimate, Task
from ..agents.charging_agent import ChargingAgent
from ..agents.safety_agent import SafetyAgent
from ..agents.single_agent import SingleAgent
from ..agents.thermal_agent import BangBangThermal, ThermalAgent, predict_temperature
from ..environment.battery_env import BatteryEnv, Scenario
from ..estimation.soc_estimator import OCVCurve, make_soc_estimator
from ..limits import current_limit_a

FAULTS = ["none", "v_bias_+0.1V", "t_bias_-5C", "t_bias_onset_-5C", "t_stuck", "t_dropout_5pct", "v_dropout_5pct"]
PROPOSERS = ["aggressive_3.5C", "conventional_1C", "independent_deadline"]


def make_stress_scenarios(n: int, seed: int) -> List[Scenario]:
    """Limit-binding regime: hot ambient, weak cooling, tight deadlines (needs ~1.2-2C), resistance up to +40%."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        amb = float(rng.uniform(35, 42))
        soh0 = float(rng.uniform(0.75, 1.0))
        out.append(Scenario(soc0=float(rng.uniform(0.1, 0.3)), loss0=1.0 - soh0, t_amb_c=amb, temp0_c=amb, target_soc=0.9,
                            deadline_s=float(rng.uniform(1500, 2400)), r0_scale=float(rng.uniform(0.9, 1.4)),
                            q_scale=float(rng.uniform(0.85, 1.05)), h_scale=float(rng.uniform(0.5, 1.0)),
                            k_r0=float(rng.uniform(1.5, 4.5)), soh0=soh0))
    return out


def apply_fault(obs: Dict[str, float], fault: str, rng: np.random.Generator, stuck_t: float) -> Dict[str, float]:
    o = dict(obs)
    if fault == "v_bias_+0.1V": o["voltage_v"] += 0.1
    elif fault == "t_bias_-5C": o["temp_c"] -= 5.0
    elif fault == "t_stuck": o["temp_c"] = stuck_t
    elif fault == "t_bias_onset_-5C" and o["time_s"] >= 300.0: o["temp_c"] -= 5.0      # bias appears after 300 s
    elif fault == "t_dropout_5pct" and rng.random() < 0.05: o["temp_c"] = np.nan
    elif fault == "v_dropout_5pct" and rng.random() < 0.05: o["voltage_v"] = np.nan
    return o


def _true_unsafe(env: BatteryEnv, i_prop: float, cfg: Dict[str, Any]) -> bool:
    if not np.isfinite(i_prop): return True
    m, L = env.model, cfg["limits"]
    v_next = m.ocv.ocv(m.soc) + i_prop * m.r0(m.temp_c) + m.v1
    return bool(i_prop > current_limit_a(cfg) + 1e-9 or v_next > L["v_max"] or (m.temp_c > L["t_max_c"] and i_prop > 0)
                or (m.soc >= L["soc_max"] and i_prop > 0) or i_prop < 0)


def _make_proposer(cfg: Dict[str, Any], name: str, sp: Optional[Dict[str, float]]) -> Tuple[Callable, Callable]:
    q = cfg["simulator"]["q0_ah"]
    if name == "aggressive_3.5C":
        return (lambda est, task: (3.5 * q, 0.0)), (lambda: None)
    if name == "conventional_1C":
        ch = ChargingAgent(cfg, c_rate=cfg["agents"]["conventional"]["c_rate"], deadline_aware=False); th = BangBangThermal(cfg)
        return (lambda est, task: (ch.propose(est, task).current_a, th.command(est.temp_meas))), th.reset
    if name == "independent_deadline":
        ch = ChargingAgent(cfg); th = ThermalAgent(cfg)
        return (lambda est, task: (ch.propose(est, task).current_a, th.propose(est).cool_cmd)), th.reset
    if name == "single_agent_tuned" and sp:
        sa = SingleAgent(cfg, sp); return (lambda est, task: sa.act(est, task)), sa.reset
    raise ValueError(name)


def run_stress_episode(cfg: Dict[str, Any], sc: Scenario, proposer: str, fault: str, safety_enabled: bool, seed: int,
                       single_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    env = BatteryEnv(cfg); obs = env.reset(seed=seed, scenario=sc); s = cfg["simulator"]; nm = cfg["agents"]["nominal_model"]
    ocv = OCVCurve(s["ocv_soc"], s["ocv_v"]); ekf = make_soc_estimator(cfg, ocv, ocv.inverse(obs["voltage_v"]), soh_est=obs["soh_est"])
    safety = SafetyAgent(cfg, enabled=safety_enabled); propose, reset = _make_proposer(cfg, proposer, single_params); reset()
    task = Task(obs["target_soc"], obs["deadline_s"]); frng = np.random.default_rng(seed + 999); stuck_t = obs["temp_c"]
    done = False; n_prop_unsafe = 0
    while not done:
        o = apply_fault(obs, fault, frng, stuck_t)
        if o["time_s"] > 0 and np.isfinite(o["voltage_v"]) and np.isfinite(o["current_a"]):
            ekf.set_capacity(s["q0_ah"] * o["soh_est"]); ekf.step(o["current_a"], o["voltage_v"], s["dt_s"])
        t_pred = predict_temperature(o["temp_c"], o["current_a"], nm, sc.t_amb_c, 0.0, s["dt_s"])
        est = StateEstimate(soc=ekf.soc, soh=o["soh_est"], v1=ekf.v1, v_meas=o["voltage_v"], i_meas=o["current_a"], temp_meas=o["temp_c"],
                            temp_pred=t_pred, time_s=o["time_s"], q_nom_ah=s["q0_ah"], r0_scale=getattr(ekf, "r0_scale", 1.0))
        i_p, cool_p = propose(est, task)
        n_prop_unsafe += int(_true_unsafe(env, i_p, cfg))
        act = safety.filter(est, i_p, cool_p)
        obs, _, term, trunc, _ = env.step([act.current_a, act.cool_cmd]); done = term or trunc
    m = env.get_metrics(); st = safety.stats(); n = max(env.step_count, 1)
    m.update({"proposer": proposer, "fault": fault, "safety": "on" if safety_enabled else "off", "seed": seed,
              "unsafe_prop_true_steps": n_prop_unsafe, "unsafe_prop_true_rate": n_prop_unsafe / n,
              "interventions": st["interventions"], "intervention_rate": st["intervention_rate"], "fallbacks": st["fallbacks"],
              "est_unsafe_prop_steps": st["unsafe_proposed"], "monitor_flags": st["monitor_flags"]})
    return m


def run_stress_grid(cfg: Dict[str, Any], scenarios: List[Scenario], proposers: List[str], faults: List[str], base_seed: int,
                    single_params: Optional[Dict[str, float]] = None, regime: str = "stress") -> pd.DataFrame:
    rows = []
    for k, sc in enumerate(scenarios):
        for p in proposers:
            for f in faults:
                for on in (True, False):                         # identical scenario + noise seed for on/off
                    r = run_stress_episode(cfg, sc, p, f, on, base_seed + k, single_params); r["scenario_idx"] = k; r["regime"] = regime; rows.append(r)
    return pd.DataFrame(rows)
