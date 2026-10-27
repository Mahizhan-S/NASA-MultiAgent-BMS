"""Health Agent (rule-based; learned policy only after the degradation model is validated).
Proposes a degradation-aware C-rate ceiling and CV voltage. Objective: slowest/gentlest charging that still meets the deadline."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
from .base import HealthProposal, StateEstimate, Task


class HealthAgent:
    def __init__(self, cfg: Dict[str, Any]):
        self.c = cfg["agents"]["health"]; self.lim = cfg["limits"]
        self.soh_eol = cfg["rul"]["soh_eol"]

    def propose(self, est: StateEstimate, task: Task) -> HealthProposal:
        c = self.c
        time_left = max(task.deadline_s - est.time_s, 1.0)
        remaining = max(task.target_soc - est.soc, 0.0)
        required_c = remaining * 3600.0 / time_left                       # C-rate needed to just meet the deadline
        health_scale = 0.5 + 0.5 * float(np.clip((est.soh - self.soh_eol) / (1 - self.soh_eol), 0, 1))
        cap = c["base_c_max"] * health_scale
        if est.temp_pred > c["t_soft_c"]:
            cap *= 0.7
        # just-in-time: do not charge faster than needed (with margin), but never below min_c
        max_c = float(np.clip(required_c * c["deadline_margin"], c["min_c"], cap))
        slack = time_left / max(remaining * 3600.0 / max(cap, 1e-6), 1.0)  # >1 => ample time
        v_cv = self.lim["v_max"] - (c["dv_health"] if slack > 1.5 else 0.0)
        return HealthProposal(max_c_rate=max_c, v_cv=v_cv, note=f"required_c={required_c:.2f},slack={slack:.2f}")
