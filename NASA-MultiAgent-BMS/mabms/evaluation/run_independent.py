"""Episode runner for: (a) conventional baseline, (b) independent specialised agents (NO coordinator).
All methods share scenarios, seeds, stopping rules and the same safety agent => comparable.
SOH estimate in simulation is a NOISY-TRUTH PLACEHOLDER (env obs['soh_est']), not a learned estimator."""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from ..agents.base import StateEstimate, Task
from ..agents.charging_agent import ChargingAgent
from ..agents.health_agent import HealthAgent
from ..agents.safety_agent import SafetyAgent
from ..agents.single_agent import SingleAgent
from ..agents.thermal_agent import BangBangThermal, ThermalAgent, predict_temperature
from ..environment.battery_env import BatteryEnv, Scenario
from ..estimation.soc_estimator import OCVCurve, make_soc_estimator

METHODS = [
    "conventional",
    "independent",
    "independent_health_cap",
    "independent_static_health_cap",
    "independent_adaptive_health",
    "adaptive_no_resistance",
    "adaptive_no_deadline_feedback",
    "adaptive_no_degradation_prediction",
]


def run_episode(cfg: Dict[str, Any], scenario: Scenario, method: str, seed: int, safety_enabled: bool = True,
                keep_log: bool = False, policy_params: Dict[str, float] | None = None,
                options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """options: {'no_thermal': bool (cooling proposal forced to 0), 'no_health': bool (health agent absent), 'health_mode': 'static'|'adaptive'}"""
    options = options or {}
    assert method in METHODS + ["single_agent"]
    env = BatteryEnv(cfg); obs = env.reset(seed=seed, scenario=scenario)
    s = cfg["simulator"]; nm = cfg["agents"]["nominal_model"]; ocv = OCVCurve(s["ocv_soc"], s["ocv_v"])
    ekf = make_soc_estimator(cfg, ocv, ocv.inverse(obs["voltage_v"]), soh_est=obs["soh_est"])
    safety = SafetyAgent(cfg, enabled=safety_enabled)
    single = None
    if method == "single_agent":
        single = SingleAgent(cfg, policy_params); charger = None; thermal = BangBangThermal(cfg)
    elif method == "conventional":
        charger = ChargingAgent(cfg, c_rate=cfg["agents"]["conventional"]["c_rate"], deadline_aware=False); thermal = BangBangThermal(cfg)
    else:
        charger = ChargingAgent(cfg); thermal = ThermalAgent(cfg)
    if charger is not None:
        charger.reset()
    health = HealthAgent(cfg); thermal.reset()
    task = Task(obs["target_soc"], obs["deadline_s"])
    h_est_module = options.get("health_estimator")
    if h_est_module is not None:
        h_est = h_est_module.reset(obs["soh_est"], getattr(ekf, "r0_scale", 1.0))
    else:
        h_est = None

    soc_err: List[float] = []; done = False; i_prev = 0.0; trace = []
    info: Dict[str, Any] = {}
    while not done:
        if h_est_module is not None:
            from ..estimation.health_interface import OracleHealthEstimator
            gt_info = {"soh": info.get("true", {}).get("soh", obs["soh_est"]), "r0_scale": getattr(ekf, "r0_scale", 1.0)} if isinstance(h_est_module, OracleHealthEstimator) else None
            h_est = h_est_module.step(obs["voltage_v"], obs["current_a"], obs["temp_c"], obs["time_s"], s["dt_s"], ground_truth=gt_info)
            soh_val = h_est.soh
            soh_sigma_val = h_est.soh_sigma
            r0_scale_val = h_est.r0_scale
        else:
            soh_val = obs["soh_est"]
            soh_sigma_val = cfg["noise"].get("soh_sigma", 0.01)
            r0_scale_val = getattr(ekf, "r0_scale", 1.0)

        ekf.set_capacity(s["q0_ah"] * soh_val)
        ekf.step(obs["current_a"] if obs["time_s"] > 0 else 0.0, obs["voltage_v"], s["dt_s"]) if obs["time_s"] > 0 else None
        t_pred = predict_temperature(obs["temp_c"], obs["current_a"], nm, scenario.t_amb_c, 0.0, s["dt_s"])
        est = StateEstimate(soc=ekf.soc, soh=soh_val, soh_sigma=soh_sigma_val, v1=ekf.v1, v_meas=obs["voltage_v"], i_meas=obs["current_a"],
                            temp_meas=obs["temp_c"], temp_pred=t_pred, time_s=obs["time_s"], q_nom_ah=s["q0_ah"], r0_scale=r0_scale_val)
        if method == "single_agent":
            i_p, cool_p = single.act(est, task)
        elif method == "conventional":
            i_p = charger.propose(est, task).current_a; cool_p = thermal.command(est.temp_meas)
        else:
            use_h = (method not in ("conventional", "independent")) and not options.get("no_health", False)
            if method == "independent_static_health_cap":
                h_mode = "static"
            elif method in ("adaptive_no_resistance", "adaptive_no_deadline_feedback", "adaptive_no_degradation_prediction"):
                h_mode = method
            else:
                h_mode = options.get("health_mode", "adaptive")
            hp = health.propose(est, task, mode=h_mode)
            cp = charger.propose(est, task, v_cv=hp.v_cv if use_h else None, max_c_rate=hp.max_c_rate if use_h else None)
            tp = thermal.propose(est)

            coord = options.get("coordinator")
            if coord is not None:
                dec = coord.coordinate(est, task, cp, hp, tp)
                i_p, cool_p = dec.current_a, dec.cool_cmd
            else:
                i_p, cool_p = cp.current_a, tp.cool_cmd
        if options.get("no_thermal", False):
            cool_p = 0.0
        act = safety.filter(est, i_p, cool_p)
        obs, _, term, trunc, info = env.step([act.current_a, act.cool_cmd]); done = term or trunc
        soc_err.append(ekf.soc - info["true"]["soc"])
        if keep_log: trace.append({"t": obs["time_s"], **info["true"], "i": act.current_a, "cool": act.cool_cmd})
    m = env.get_metrics()
    m.update(safety.stats()); m.update({"method": method, "seed": seed, "safety_enabled": safety_enabled,
                                        "soc_est_mae": float(np.mean(np.abs(soc_err))), "soc_est_final_err": float(soc_err[-1]), "soc_est_rmse": float(np.sqrt(np.mean(np.square(soc_err))))})
    if keep_log:
        m["_trace"] = pd.DataFrame(trace); m["_safety_log"] = pd.DataFrame(safety.log)
    return m


def run_benchmark(cfg: Dict[str, Any], scenarios: List[Scenario], methods: List[str], base_seed: int,
                  safety_enabled: bool = True, n_seeds: int = 1, policy_params: Dict[str, float] | None = None,
                  options: Dict[str, Any] | None = None, label: str | None = None) -> pd.DataFrame:
    rows = []
    for k, sc in enumerate(scenarios):
        for j in range(n_seeds):
            for meth in methods:                   # same scenario & noise seed for every method
                r = run_episode(cfg, sc, meth, seed=base_seed + 1000 * j + k, safety_enabled=safety_enabled,
                                policy_params=policy_params, options=options)
                r["scenario_idx"] = k; r["noise_seed_idx"] = j
                if label: r["method"] = label
                rows.append(r)
    return pd.DataFrame(rows)
