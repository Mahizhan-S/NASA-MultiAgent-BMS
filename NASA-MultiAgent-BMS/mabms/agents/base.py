"""Shared dataclasses for agent inputs/outputs."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np


@dataclass
class StateEstimate:
    soc: float; soh: float; v1: float
    v_meas: float; i_meas: float; temp_meas: float; temp_pred: float; time_s: float
    q_nom_ah: float = 2.0
    r0_scale: float = 1.0      # estimated R0 / nominal R0 (1.0 when the estimator does not adapt it)
    soh_sigma: float = 0.02    # health estimation uncertainty (standard deviation)

    def finite(self) -> bool:
        return all(np.isfinite(x) for x in (self.soc, self.soh, self.v1, self.v_meas, self.i_meas, self.temp_meas, self.temp_pred))


@dataclass
class Task:
    target_soc: float; deadline_s: float


@dataclass
class HealthProposal:
    max_c_rate: float
    v_cv: float
    note: str = ""
    feasibility_status: str = "FEASIBLE"   # "FEASIBLE", "AT_RISK", "INFEASIBLE"
    predicted_dloss: float = 0.0
    i_required_a: float = 0.0
    is_fallback: bool = False


@dataclass
class ThermalProposal:
    cool_cmd: float; temp_setpoint_c: float


@dataclass
class ChargingProposal:
    current_a: float; mode: str


@dataclass
class SafeAction:
    current_a: float; cool_cmd: float; intervened: bool = False; reasons: List[str] = field(default_factory=list)
    fallback: bool = False
