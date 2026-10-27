"""Health Agent (rule-based; learned policy only after the degradation model is validated).
Proposes a degradation-aware C-rate ceiling and CV voltage. Objective: slowest/gentlest charging that still meets the deadline."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
from .base import HealthProposal, StateEstimate, Task


class HealthAgent:
    def __init__(self, cfg: Dict[str, Any], mode: str = "adaptive"):
        self.c = cfg["agents"]["health"]
        self.lim = cfg["limits"]
        self.soh_eol = cfg["rul"]["soh_eol"]
        self.t_ref = cfg["agents"]["nominal_model"].get("t_ref_c", 25.0)
        self.default_mode = cfg["agents"]["health"].get("mode", mode)

    def propose(self, est: StateEstimate, task: Task, mode: str | None = None) -> HealthProposal:
        use_mode = mode or self.default_mode
        c = self.c
        time_left = max(task.deadline_s - est.time_s, 1.0)
        remaining = max(task.target_soc - est.soc, 0.0)
        required_c = remaining * 3600.0 / time_left
        deadline_c = required_c * c.get("deadline_margin", 1.15)

        if use_mode == "static":
            # Original baseline fixed rule-based health cap
            health_scale = 0.5 + 0.5 * float(np.clip((est.soh - self.soh_eol) / max(1.0 - self.soh_eol, 1e-3), 0.0, 1.0))
            cap = c["base_c_max"] * health_scale
            if est.temp_pred > c.get("t_soft_c", 38.0):
                cap *= 0.7
            max_c = float(np.clip(deadline_c, c.get("min_c", 0.3), cap))
            slack = time_left / max(remaining * 3600.0 / max(cap, 1e-6), 1.0)
            v_cv = self.lim["v_max"] - (c.get("dv_health", 0.05) if slack > 1.5 else 0.0)
            return HealthProposal(max_c_rate=max_c, v_cv=v_cv, note=f"mode=static,req_c={required_c:.2f},cap={cap:.2f},slack={slack:.2f}")

        # Adaptive Health Mode:
        soh_norm = float(np.clip((est.soh - self.soh_eol) / max(1.0 - self.soh_eol, 1e-3), 0.0, 1.0))
        f_soh = 0.6 + 0.4 * soh_norm
        r0_scale = max(getattr(est, "r0_scale", 1.0), 1.0)
        f_rho = 1.0 / np.sqrt(r0_scale)
        temp_val = est.temp_pred if np.isfinite(est.temp_pred) else est.temp_meas
        t_soft = c.get("t_soft_c", 38.0)
        f_temp = float(np.exp(-0.08 * (temp_val - t_soft))) if temp_val > t_soft else 1.0
        f_soc = 1.0 - 0.35 * (((est.soc - 0.8) / 0.2) ** 2) if est.soc > 0.8 else 1.0
        c_health_max = c["base_c_max"] * f_soh * f_rho * f_temp * f_soc
        max_c = float(np.clip(max(c_health_max, deadline_c), c.get("min_c", 0.3), self.lim["i_max_c"]))
        slack = time_left / max(remaining * 3600.0 / max(c_health_max, 1e-6), 1.0)
        v_cv = self.lim["v_max"] - (c.get("dv_health", 0.05) if slack > 1.3 else 0.0)
        note = f"mode=adaptive,req_c={required_c:.2f},c_health_max={c_health_max:.2f},f_rho={f_rho:.2f},slack={slack:.2f}"
        return HealthProposal(max_c_rate=max_c, v_cv=v_cv, note=note)
