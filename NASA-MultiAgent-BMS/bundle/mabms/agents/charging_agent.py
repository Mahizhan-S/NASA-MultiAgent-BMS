"""Charging Agent: model-based CC-CV with deadline pressure (conventional baseline first; RL later).
CV current limit comes from the EKF-informed model: i_cv = (v_cv - OCV(soc) - v1)/R0_nominal."""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from ..estimation.soc_estimator import OCVCurve
from ..limits import current_limit_a
from .base import ChargingProposal, StateEstimate, Task


class ChargingAgent:
    def __init__(self, cfg: Dict[str, Any], c_rate: Optional[float] = None, deadline_aware: bool = True):
        self.cfg = cfg; self.lim = cfg["limits"]; self.i_abs = current_limit_a(cfg)
        self.c_rate = c_rate if c_rate is not None else cfg["agents"]["charging"]["c_rate"]
        self.margin = cfg["agents"]["charging"]["deadline_margin"]; self.deadline_aware = deadline_aware
        s = cfg["simulator"]; self.ocv = OCVCurve(s["ocv_soc"], s["ocv_v"])
        self.r0 = cfg["agents"]["nominal_model"]["r0_ohm"]

    def propose(self, est: StateEstimate, task: Task, v_cv: Optional[float] = None, max_c_rate: Optional[float] = None) -> ChargingProposal:
        if est.soc >= task.target_soc:
            return ChargingProposal(0.0, "done")
        q = est.q_nom_ah * est.soh
        c = self.c_rate
        if self.deadline_aware:
            required_c = (task.target_soc - est.soc) * 3600.0 / max(task.deadline_s - est.time_s, 1.0)
            c = max(c, required_c * self.margin)
        c = min(c, self.lim["i_max_c"])
        if max_c_rate is not None:
            c = min(c, max_c_rate)
        i_cc = c * q
        v_lim = v_cv if v_cv is not None else self.lim["v_max"]
        i_cv = max((v_lim - self.ocv.ocv(est.soc) - est.v1) / self.r0, 0.0)
        i_cc = min(i_cc, self.i_abs)
        return ChargingProposal(float(min(i_cc, i_cv)), "CC" if i_cc <= i_cv else "CV")
