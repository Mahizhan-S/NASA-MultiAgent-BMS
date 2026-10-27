"""Health Agent (rule-based; learned policy only after the degradation model is validated).
Proposes a degradation-aware C-rate ceiling and CV voltage. Objective: slowest/gentlest charging that still meets the deadline."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
from .base import HealthProposal, StateEstimate, Task


class HealthAgent:
    def __init__(self, cfg: Dict[str, Any]):
        self.c = cfg["agents"]["health"]
        self.lim = cfg["limits"]
        self.soh_eol = cfg["rul"]["soh_eol"]
        self.t_ref = cfg["agents"]["nominal_model"].get("t_ref_c", 25.0)

    def propose(self, est: StateEstimate, task: Task) -> HealthProposal:
        c = self.c
        time_left = max(task.deadline_s - est.time_s, 1.0)
        remaining = max(task.target_soc - est.soc, 0.0)

        # 1. Deadline feasibility floor: minimum C-rate required to meet the deadline
        required_c = remaining * 3600.0 / time_left
        deadline_c = required_c * c.get("deadline_margin", 1.15)

        # 2. Adaptive degradation factors from EKF & battery condition:
        # (a) SOH capacity margin factor
        soh_norm = float(np.clip((est.soh - self.soh_eol) / max(1.0 - self.soh_eol, 1e-3), 0.0, 1.0))
        f_soh = 0.6 + 0.4 * soh_norm

        # (b) Internal resistance factor: as R0 grows (r0_scale > 1), Joule heating (I^2*R) increases;
        # derate current proportionally to 1/sqrt(r0_scale) to keep overpotential and heating bounded
        r0_scale = max(getattr(est, "r0_scale", 1.0), 1.0)
        f_rho = 1.0 / np.sqrt(r0_scale)

        # (c) Temperature derating: smooth Arrhenius degradation penalty above reference temp
        temp_val = est.temp_pred if np.isfinite(est.temp_pred) else est.temp_meas
        t_soft = c.get("t_soft_c", 38.0)
        if temp_val > t_soft:
            f_temp = float(np.exp(-0.08 * (temp_val - t_soft)))
        else:
            f_temp = 1.0

        # (d) High-SOC stress factor: quadratic penalty above 80% SOC to suppress lithium plating
        if est.soc > 0.8:
            soc_excess = (est.soc - 0.8) / 0.2
            f_soc = 1.0 - 0.35 * (soc_excess ** 2)
        else:
            f_soc = 1.0

        # Combined adaptive health ceiling
        c_health_max = c["base_c_max"] * f_soh * f_rho * f_temp * f_soc

        # 3. Deadline-Health reconciliation:
        # The health ceiling protects the battery by derating current when aging, resistance,
        # temperature, or high SOC increases stress. However, to guarantee deadline feasibility
        # and prevent current starvation of aged cells, the ceiling expands if the deadline floor
        # (deadline_c) requires it, bounded by the hardware physical limit.
        max_c = float(np.clip(max(c_health_max, deadline_c), c.get("min_c", 0.3), self.lim["i_max_c"]))

        slack = time_left / max(remaining * 3600.0 / max(c_health_max, 1e-6), 1.0)
        # Drop CV setpoint when ample time exists to reduce calendar/SEI degradation at high voltage
        v_cv = self.lim["v_max"] - (c.get("dv_health", 0.05) if slack > 1.3 else 0.0)

        note = f"req_c={required_c:.2f},c_health_max={c_health_max:.2f},f_rho={f_rho:.2f},slack={slack:.2f}"
        return HealthProposal(max_c_rate=max_c, v_cv=v_cv, note=note)
