"""Thermal Agent: PI feedback controller on predicted temperature (default) + bang-bang baseline.
Predictor uses a NOMINAL lumped model (may be mismatched to the plant). No learned policy until a realistic plant is validated."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
from .base import StateEstimate, ThermalProposal


def predict_temperature(T: float, i_a: float, nm: Dict[str, float], t_amb: float, cool: float, dt: float, steps: int = 6) -> float:
    for _ in range(steps):
        heat = i_a ** 2 * (nm["r0_ohm"] + nm["r1_ohm"])
        T += dt * (heat - nm["h_w_per_k"] * (T - t_amb) - cool * nm["cool_max_w"]) / nm["c_th_j_per_k"]
    return T


class ThermalAgent:
    def __init__(self, cfg: Dict[str, Any]):
        self.k = cfg["agents"]["thermal"]; self.i = 0.0; self.dt = cfg["simulator"]["dt_s"]
        self.setpoint = self.k["setpoint_c"]

    def reset(self) -> None:
        self.i = 0.0

    def propose(self, est: StateEstimate) -> ThermalProposal:
        e = est.temp_pred - self.setpoint
        u_raw = self.k["kp"] * e + self.k["ki"] * self.i
        u = float(np.clip(u_raw, 0.0, 1.0))
        if (u_raw == u) or (u_raw > 1 and e < 0) or (u_raw < 0 and e > 0):   # anti-windup
            self.i += e * self.dt
        return ThermalProposal(cool_cmd=u, temp_setpoint_c=self.setpoint)


class BangBangThermal:
    def __init__(self, cfg: Dict[str, Any]):
        self.on, self.off = cfg["agents"]["conventional"]["cool_on_c"], cfg["agents"]["conventional"]["cool_off_c"]
        self.state = 0.0

    def reset(self) -> None:
        self.state = 0.0

    def command(self, temp_c: float) -> float:
        if temp_c > self.on: self.state = 1.0
        elif temp_c < self.off: self.state = 0.0
        return self.state
