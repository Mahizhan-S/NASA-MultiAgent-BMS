"""Charging + thermal-control environment. Interface: reset(seed, scenario), observe(), step(action),
get_metrics(), terminate(). Action: [current_a (>=0 charge), cool_cmd in [0,1]].
True state is only exposed through info['true'] for diagnostics; agents must use observe() (noisy)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .battery_model import BatteryModel, ModelParams
from ..limits import current_limit_a


@dataclass
class Scenario:
    soc0: float = 0.2; loss0: float = 0.0; t_amb_c: float = 30.0; temp0_c: float = 30.0
    target_soc: float = 0.9; deadline_s: float = 3600.0
    r0_scale: float = 1.0; q_scale: float = 1.0; h_scale: float = 1.0
    v_sigma: Optional[float] = None; i_sigma: Optional[float] = None; t_sigma: Optional[float] = None


def make_scenarios(n: int, seed: int, regime: str = "train") -> List[Scenario]:
    """regime='train'/'val': narrow ranges. regime='test_unseen': wider parameter shifts + higher ambient (distribution shift)."""
    rng = np.random.default_rng(seed)
    wide = regime == "test_unseen"
    out = []
    for _ in range(n):
        out.append(Scenario(
            soc0=float(rng.uniform(0.1, 0.4)), loss0=float(rng.uniform(0.0, 0.25 if wide else 0.15)),
            t_amb_c=float(rng.uniform(28, 38) if wide else rng.uniform(25, 33)), temp0_c=0.0,
            target_soc=float(rng.choice([0.8, 0.9])), deadline_s=float(rng.uniform(2400, 5400)),
            r0_scale=float(rng.uniform(0.7, 1.5) if wide else rng.uniform(0.9, 1.1)),
            q_scale=float(rng.uniform(0.85, 1.1) if wide else rng.uniform(0.95, 1.05)),
            h_scale=float(rng.uniform(0.6, 1.2) if wide else rng.uniform(0.9, 1.1))))
        out[-1].temp0_c = out[-1].t_amb_c
    return out


class BatteryEnv:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg; self.dt = cfg["simulator"]["dt_s"]; self.lim = cfg["limits"]; self.noise = cfg["noise"]
        self.base = ModelParams.from_config(cfg)
        self.rng = np.random.default_rng(0); self.model: Optional[BatteryModel] = None
        self.sc = Scenario(); self.done = True

    # ---- interface ----
    def reset(self, seed: Optional[int] = None, scenario: Optional[Scenario] = None) -> Dict[str, float]:
        self.rng = np.random.default_rng(seed)
        self.sc = scenario or Scenario()
        p = ModelParams(**{**self.base.__dict__})
        p.r0_ohm *= self.sc.r0_scale; p.q0_ah *= self.sc.q_scale; p.h_w_per_k *= self.sc.h_scale
        self.model = BatteryModel(p, self.sc.soc0, self.sc.loss0, self.sc.temp0_c, self.sc.t_amb_c)
        self.t = 0.0; self.i_applied = 0.0; self.done = False; self.step_count = 0
        self.m = dict(time_to_target=None, energy_in_wh=0.0, cooling_energy_wh=0.0, peak_temp_c=self.model.temp_c,
                      steps_v_viol=0, steps_t_viol=0, steps_i_viol=0, steps_soc_viol=0, hard_violation=False,
                      temp_err_sum=0.0, loss_start=self.model.loss)
        self.m["target_reached"] = False; self.m["max_v"] = 0.0; self.m["soc_start"] = self.model.soc
        return self.observe()

    def observe(self) -> Dict[str, float]:
        m = self.model
        sv = self.sc.v_sigma if self.sc.v_sigma is not None else self.noise["v_sigma"]
        si = self.sc.i_sigma if self.sc.i_sigma is not None else self.noise["i_sigma"]
        st = self.sc.t_sigma if self.sc.t_sigma is not None else self.noise["t_sigma"]
        return {"time_s": self.t, "voltage_v": m.last_v + self.rng.normal(0, sv),
                "current_a": self.i_applied + self.rng.normal(0, si), "temp_c": m.temp_c + self.rng.normal(0, st),
                "target_soc": self.sc.target_soc, "deadline_s": self.sc.deadline_s, "dt_s": self.dt,
                "soh_est": float(np.clip(m.soh + self.rng.normal(0, self.noise["soh_sigma"]), 0, 1.2))}

    def step(self, action) -> Tuple[Dict[str, float], float, bool, bool, Dict[str, Any]]:
        assert not self.done, "call reset() first"
        i_cmd, cool = float(action[0]), float(action[1])
        if not np.isfinite(i_cmd) or not np.isfinite(cool):   # invalid action -> safe no-op
            i_cmd, cool = 0.0, 1.0
        q0 = self.model.p.q0_ah
        i_hw = float(np.clip(i_cmd, 0.0, current_limit_a(self.cfg) * 1.5))   # physical hardware ceiling (150% of limit)
        out = self.model.step(i_hw, cool, self.dt)
        self.i_applied = i_hw; self.t += self.dt; self.step_count += 1
        v, T, soc = out["voltage"], self.model.temp_c, self.model.soc
        L = self.lim; m = self.m
        v_viol = v > L["v_max"]; t_viol = T > L["t_max_c"]; i_viol = i_hw > current_limit_a(self.cfg) + 1e-9; s_viol = soc > L["soc_max"]
        m["steps_v_viol"] += int(v_viol); m["steps_t_viol"] += int(t_viol)
        m["steps_i_viol"] += int(i_viol); m["steps_soc_viol"] += int(s_viol)
        m["energy_in_wh"] += v * i_hw * self.dt / 3600.0; m["cooling_energy_wh"] += out["cool_energy_wh"]
        m["peak_temp_c"] = max(m["peak_temp_c"], T); m["max_v"] = max(m["max_v"], v); m["temp_err_sum"] += abs(T - self.cfg["simulator"]["t_target_c"])
        hard = v > L["v_hard"] or T > L["t_hard_c"]
        m["hard_violation"] |= hard
        terminated = False
        if soc >= self.sc.target_soc - self.cfg["simulator"].get("target_tolerance", 0.0) and not m["target_reached"]:
            m["target_reached"] = True; m["time_to_target"] = self.t; terminated = True
        if hard and self.cfg["simulator"]["terminate_on_hard_violation"]:
            terminated = True
        truncated = (not terminated) and self.t >= self.sc.deadline_s
        self.done = terminated or truncated
        violation = float(v_viol or t_viol or i_viol or s_viol)
        reward = -(self.dt / self.sc.deadline_s) - 1e3 * out["dloss"] - out["cool_energy_wh"] - 10.0 * violation
        info = {"true": {"soc": soc, "soh": self.model.soh, "temp_c": T, "voltage": v}, "violation": bool(violation), "hard": bool(hard)}
        return self.observe(), float(reward), terminated, truncated, info

    def get_metrics(self) -> Dict[str, Any]:
        m, n = self.m, max(self.step_count, 1)
        return {"time_to_target_s": m["time_to_target"], "target_reached": m["target_reached"],
                "time_to_target_penalized_s": (m["time_to_target"] if m["target_reached"] else self.sc.deadline_s),
                "target_soc": self.sc.target_soc, "deadline_s": self.sc.deadline_s,
                "shortfall": max(0.0, self.sc.target_soc - self.model.soc),
                "is_near_miss": bool((not m["target_reached"]) and (self.sc.target_soc - self.model.soc) < 0.010),
                "is_starvation": bool((not m["target_reached"]) and (self.sc.target_soc - self.model.soc) >= 0.010),
                "final_soc_true": self.model.soc, "energy_in_wh": m["energy_in_wh"],
                "peak_temp_c": m["peak_temp_c"], "max_v_overshoot_v": max(m["max_v"] - self.lim["v_max"], 0.0), "temp_track_mae_c": m["temp_err_sum"] / n,
                "time_over_tmax_s": m["steps_t_viol"] * self.dt, "cooling_energy_wh": m["cooling_energy_wh"],
                "capacity_loss": self.model.loss - m["loss_start"],
                "soc_gained": self.model.soc - m["soc_start"],
                "loss_per_soc_gained": (self.model.loss - m["loss_start"]) / max(self.model.soc - m["soc_start"], 1e-3),
                "loss_per_kwh": (self.model.loss - m["loss_start"]) / max(m["energy_in_wh"] / 1000.0, 1e-6),
                "sim_violations_v": m["steps_v_viol"], "sim_violations_t": m["steps_t_viol"],
                "sim_violations_i": m["steps_i_viol"], "sim_violations_soc": m["steps_soc_viol"],
                "unsafe_executed_steps": m["steps_v_viol"] + m["steps_t_viol"] + m["steps_i_viol"] + m["steps_soc_viol"],
                "hard_violation": m["hard_violation"], "steps": self.step_count}

    def terminate(self) -> Dict[str, Any]:
        self.done = True
        return self.get_metrics()
