"""Charging Agent: model-based CC-CV with deadline pressure (conventional baseline first; RL later).
CV current limit comes from the EKF-informed model: i_cv = (v_cv - OCV(soc) - v1)/R0_nominal.

Phase I Hardening:
1. Strict input validation rejecting NaN, negative capacity, out-of-bounds target SOC, invalid states.
2. Non-negative current proposal enforcement (I_prop >= 0).
3. Explicit reset() lifecycle method for episode reuse.
4. Coulombic efficiency in required C-rate deadline calculation.
5. Deterministic fallback proposal on invalid inputs.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from ..estimation.soc_estimator import OCVCurve
from ..limits import current_limit_a
from .base import ChargingProposal, StateEstimate, Task


class ChargingAgent:
    def __init__(self, cfg: Dict[str, Any], c_rate: Optional[float] = None, deadline_aware: bool = True):
        self.cfg = cfg
        self.lim = cfg["limits"]
        self.i_abs = current_limit_a(cfg)
        self.c_rate = c_rate if c_rate is not None else cfg["agents"]["charging"]["c_rate"]
        self.margin = cfg["agents"]["charging"]["deadline_margin"]
        self.deadline_aware = deadline_aware
        ch = cfg["agents"]["charging"]
        self.stop_margin = ch.get("soc_stop_margin", 0.0)
        self.hyst = ch.get("restart_hysteresis", 0.0)
        self._stopped = False
        s = cfg["simulator"]
        self.ocv = OCVCurve(s["ocv_soc"], s["ocv_v"])
        self.r0 = cfg["agents"]["nominal_model"]["r0_ohm"]
        self.coulombic_eff = float(s.get("coulombic_eff", 0.995))

    def reset(self) -> None:
        """Reset internal episode-specific latch state."""
        self._stopped = False

    def _validate_inputs(self, est: StateEstimate, task: Task,
                         v_cv: Optional[float] = None,
                         max_c_rate: Optional[float] = None) -> bool:
        """Validate state, task, and optional ceilings. Reject invalid/corrupted inputs."""
        # 1. StateEstimate finite check and validity flag
        if not est.finite() or not getattr(est, "is_valid", True):
            return False

        # 2. Physical state bounds
        if not (-0.1 <= est.soc <= 1.2):
            return False
        if not (0.2 < est.soh <= 1.5):
            return False
        if not (est.q_nom_ah > 0.1):
            return False
        if not (est.r0_scale > 0.05):
            return False
        if not (1.0 <= est.v_meas <= 5.5):
            return False
        if not (-50.0 <= est.temp_meas <= 120.0):
            return False

        # 3. Task bounds
        if not (np.isfinite(task.target_soc) and np.isfinite(task.deadline_s)):
            return False
        if not (0.0 < task.target_soc <= 1.0):
            return False
        if task.deadline_s <= 0.0 or est.time_s < 0.0:
            return False

        # 4. Optional ceilings
        if max_c_rate is not None:
            if not np.isfinite(max_c_rate) or max_c_rate < 0.0:
                return False
        if v_cv is not None:
            if not np.isfinite(v_cv) or not (3.0 <= v_cv <= 4.35):
                return False

        return True

    def propose(self, est: StateEstimate, task: Task,
                v_cv: Optional[float] = None,
                max_c_rate: Optional[float] = None) -> ChargingProposal:
        # 1. Input validation gate
        if not self._validate_inputs(est, task, v_cv, max_c_rate):
            return ChargingProposal(
                current_a=0.0,
                mode="fallback:invalid_inputs",
                is_fallback=True,
                is_valid=False,
                required_c=0.0
            )

        # 2. Stop latch and hysteresis check
        tgt = task.target_soc + self.stop_margin
        if est.soc >= tgt:
            self._stopped = True
        elif self._stopped and est.soc < task.target_soc - self.hyst:
            self._stopped = False  # resume only if clearly below target

        if self._stopped:
            return ChargingProposal(0.0, "done", is_fallback=False, is_valid=True, required_c=0.0)

        # 3. CC Current Demand Calculation
        q = est.q_nom_ah * est.soh
        c = self.c_rate
        required_c = 0.0

        if self.deadline_aware:
            delta_soc = max(tgt - est.soc, 0.0)
            time_left_s = max(task.deadline_s - est.time_s, 1.0)
            # Incorporate configured coulombic efficiency
            eff = max(self.coulombic_eff, 0.5)
            required_c = (delta_soc / eff) * 3600.0 / time_left_s
            c = max(c, required_c * self.margin)

        # Apply C-rate hardware limit
        c = min(c, self.lim["i_max_c"])

        # Apply health agent ceiling if provided
        if max_c_rate is not None:
            c = min(c, max(0.0, float(max_c_rate)))

        # Ensure C-rate is strictly non-negative
        c = max(0.0, c)
        i_cc = c * q

        # 4. CV Current Limiting Calculation
        v_lim = v_cv if v_cv is not None else self.lim["v_max"]
        r0_eff = max(self.r0 * max(est.r0_scale, 0.05), 1e-4)
        i_cv = max((v_lim - self.ocv.ocv(est.soc) - est.v1) / r0_eff, 0.0)

        # Absolute hardware current clamp
        i_cc = min(i_cc, self.i_abs)

        # Strictly non-negative, feasible candidate current
        i_cand = max(0.0, min(i_cc, i_cv))

        return ChargingProposal(
            current_a=float(i_cand),
            mode="CC" if i_cc <= i_cv else "CV",
            is_fallback=False,
            is_valid=True,
            required_c=float(required_c)
        )
