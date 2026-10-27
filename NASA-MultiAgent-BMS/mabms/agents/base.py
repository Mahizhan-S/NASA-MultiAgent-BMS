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
    is_valid: bool = True
    timestamp_s: float = 0.0

    def __post_init__(self):
        if self.timestamp_s == 0.0:
            self.timestamp_s = self.time_s
        if not self.finite() or not (0.2 < self.soh <= 1.5):
            self.is_valid = False

    def finite(self) -> bool:
        return all(np.isfinite(x) for x in (self.soc, self.soh, self.v1, self.v_meas, self.i_meas, self.temp_meas, self.temp_pred))


@dataclass
class Task:
    target_soc: float; deadline_s: float


@dataclass
class HealthProposal:
    max_c_rate: float
    v_cv: float = 4.20
    note: str = ""
    feasibility_status: str = "FEASIBLE"   # "FEASIBLE", "AT_RISK", "INFEASIBLE"
    predicted_dloss: float = 0.0
    i_required_a: float = 0.0
    is_fallback: bool = False
    is_valid: bool = True
    timestamp_s: float = 0.0

    @property
    def i_health_max(self) -> float:
        return float(self.max_c_rate * 2.45) if self.max_c_rate is not None else 0.0



@dataclass
class ThermalProposal:
    cool_cmd: float; temp_setpoint_c: float = 35.0


@dataclass
class ChargingProposal:
    current_a: float
    mode: str
    is_fallback: bool = False
    is_valid: bool = True
    required_c: float = 0.0



@dataclass
class SafeAction:
    current_a: float; cool_cmd: float; intervened: bool = False; reasons: List[str] = field(default_factory=list)
    fallback: bool = False
