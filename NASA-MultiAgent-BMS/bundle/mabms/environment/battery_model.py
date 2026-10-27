"""Per-cell-equivalent surrogate: 1-RC ECM + lumped thermal + empirical degradation.
ALL PARAMETERS ARE ASSUMPTIONS until calibrated (see scripts/04_validate_sim.py). Not a NASA measurement.
Convention: current > 0 is charge."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np
from ..estimation.soc_estimator import OCVCurve

K0 = 273.15


@dataclass
class ModelParams:
    q0_ah: float = 2.0; coulombic_eff: float = 0.995
    r0_ohm: float = 0.08; r1_ohm: float = 0.04; c1_f: float = 2000.0
    r_ea_k: float = 2000.0; t_ref_c: float = 25.0
    c_th: float = 45.0; h_w_per_k: float = 0.05; cool_max_w: float = 1.0; cool_elec_w: float = 0.5
    k_deg: float = 2e-4; ea_deg_k: float = 4000.0; c_rate_coeff: float = 0.5; soc_stress: float = 1.0
    ocv_soc: tuple = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
    ocv_v: tuple = (3.20, 3.45, 3.58, 3.68, 3.82, 4.00, 4.10, 4.20)

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "ModelParams":
        s = cfg["simulator"]; th = s["thermal"]; dg = s["degradation"]
        return ModelParams(q0_ah=s["q0_ah"], coulombic_eff=s["coulombic_eff"], r0_ohm=s["r0_ohm"], r1_ohm=s["r1_ohm"],
                           c1_f=s["c1_f"], r_ea_k=s["r_ea_k"], t_ref_c=s["t_ref_c"], c_th=th["c_th_j_per_k"],
                           h_w_per_k=th["h_w_per_k"], cool_max_w=th["cool_max_w"], cool_elec_w=th["cool_elec_w"],
                           k_deg=dg["k_per_eq_cycle"], ea_deg_k=dg["ea_over_r_k"], c_rate_coeff=dg["c_rate_coeff"],
                           soc_stress=dg["soc_stress_coeff"], ocv_soc=tuple(s["ocv_soc"]), ocv_v=tuple(s["ocv_v"]))


class BatteryModel:
    def __init__(self, p: ModelParams, soc0=0.2, loss0=0.0, temp0_c=30.0, t_amb_c=30.0):
        self.p = p; self.ocv = OCVCurve(p.ocv_soc, p.ocv_v)
        self.soc, self.v1, self.temp_c, self.loss, self.t_amb_c = soc0, 0.0, temp0_c, loss0, t_amb_c
        self.last_v = self.ocv.ocv(soc0)

    @property
    def capacity_ah(self) -> float:
        return self.p.q0_ah * (1.0 - self.loss)

    @property
    def soh(self) -> float:
        return 1.0 - self.loss

    def r0(self, temp_c: float) -> float:
        return self.p.r0_ohm * np.exp(self.p.r_ea_k * (1 / (temp_c + K0) - 1 / (self.p.t_ref_c + K0)))

    def terminal_voltage(self, i_a: float) -> float:
        return self.ocv.ocv(self.soc) + i_a * self.r0(self.temp_c) + self.v1

    def step(self, i_a: float, cool_cmd: float, dt: float) -> Dict[str, float]:
        p = self.p
        cool_cmd = float(np.clip(cool_cmd, 0.0, 1.0))
        eta = p.coulombic_eff if i_a > 0 else 1.0
        self.soc += eta * i_a * dt / (3600.0 * self.capacity_ah)
        a = np.exp(-dt / (p.r1_ohm * p.c1_f))
        self.v1 = a * self.v1 + p.r1_ohm * (1 - a) * i_a
        r0 = self.r0(self.temp_c)
        heat_w = i_a ** 2 * (r0 + p.r1_ohm)
        dT = (heat_w - p.h_w_per_k * (self.temp_c - self.t_amb_c) - cool_cmd * p.cool_max_w) / p.c_th
        self.temp_c += dT * dt
        # degradation surrogate: loss per Ah throughput scaled by T, C-rate and high-SOC stress
        c_rate = abs(i_a) / p.q0_ah
        f_T = np.exp(p.ea_deg_k * (1 / (p.t_ref_c + K0) - 1 / (self.temp_c + K0)))
        f_C = 1.0 + p.c_rate_coeff * max(c_rate - 0.5, 0.0)
        f_S = 1.0 + p.soc_stress * (max(self.soc - 0.8, 0.0) / 0.2) ** 2
        dloss = p.k_deg * (abs(i_a) * dt / 3600.0 / p.q0_ah) * f_T * f_C * f_S
        self.loss += dloss
        v = self.terminal_voltage(i_a)
        self.last_v = v
        return {"voltage": v, "heat_w": heat_w, "dloss": dloss, "cool_energy_wh": cool_cmd * p.cool_elec_w * dt / 3600.0}
