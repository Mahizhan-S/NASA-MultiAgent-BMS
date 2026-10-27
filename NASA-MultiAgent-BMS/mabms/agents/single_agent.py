"""Baseline 2: ONE controller, ONE parameter vector, outputs both current and cooling, tuned on a shared objective J.
Parametric policy + random search (no RL dependency). An RL version (PPO/SAC) can replace `act` later with the same interface."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from ..estimation.soc_estimator import OCVCurve
from ..limits import current_limit_a
from .base import StateEstimate, Task
from .thermal_agent import predict_temperature

BOUNDS = {"c_rate": (0.3, 2.0), "v_cv_offset": (0.0, 0.08), "kp": (0.05, 1.0), "setpoint_c": (28.0, 40.0), "deadline_gain": (0.8, 1.5)}


def sample_params(rng: np.random.Generator) -> Dict[str, float]:
    return {k: float(rng.uniform(*b)) for k, b in BOUNDS.items()}


class SingleAgent:
    def __init__(self, cfg: Dict[str, Any], params: Dict[str, float]):
        self.cfg, self.p, self.lim = cfg, params, cfg["limits"]; self.i_abs = current_limit_a(cfg)
        s = cfg["simulator"]; self.ocv = OCVCurve(s["ocv_soc"], s["ocv_v"]); self.r0 = cfg["agents"]["nominal_model"]["r0_ohm"]
        ch = cfg["agents"]["charging"]; self.stop_margin = ch.get("soc_stop_margin", 0.0); self.hyst = ch.get("restart_hysteresis", 0.0); self._stopped = False

    def reset(self) -> None: self._stopped = False

    def act(self, est: StateEstimate, task: Task) -> Tuple[float, float]:
        tgt = task.target_soc + self.stop_margin
        if est.soc >= tgt: self._stopped = True
        elif self._stopped and est.soc < task.target_soc - self.hyst: self._stopped = False
        if self._stopped:
            return 0.0, 1.0 if est.temp_pred > self.p["setpoint_c"] else 0.0
        q = est.q_nom_ah * est.soh
        req = (tgt - est.soc) * 3600.0 / max(task.deadline_s - est.time_s, 1.0)
        c = min(max(self.p["c_rate"], 0.0 if self.p["deadline_gain"] <= 0 else req * self.p["deadline_gain"]), self.lim["i_max_c"])
        v_lim = self.lim["v_max"] - self.p["v_cv_offset"]
        i_cv = max((v_lim - self.ocv.ocv(est.soc) - est.v1) / (self.r0 * est.r0_scale), 0.0)
        cool = float(np.clip(self.p["kp"] * (est.temp_pred - self.p["setpoint_c"]), 0.0, 1.0))
        return float(min(c * q, i_cv, self.i_abs)), cool
