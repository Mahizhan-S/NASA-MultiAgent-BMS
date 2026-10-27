"""Multi-Agent Coordinator Architectures (Phase H).
Implements and evaluates alternative proposal fusion mechanisms:
1. MinimumRuleCoordinator: Baseline rule taking I = min(I_speed, I_health).
2. DynamicPriorityCoordinator: State-dependent priority switching based on deadline urgency and thermal status.
3. ConstrainedQPCoordinator: Analytic convex quadratic program balancing speed, health, and thermal loss under physical feasibility bounds.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

from .base import ChargingProposal, HealthProposal, SafeAction, StateEstimate, Task, ThermalProposal


@dataclass
class CoordinatorDecision:
    current_a: float
    cool_cmd: float
    coordinator_mode: str
    arbitration_reason: str
    weights: Optional[Dict[str, float]] = None


class BaseCoordinator:
    """Abstract interface for multi-agent coordinator."""
    def coordinate(self, est: StateEstimate, task: Task,
                   cp: ChargingProposal, hp: HealthProposal,
                   tp: ThermalProposal) -> CoordinatorDecision:
        raise NotImplementedError


class MinimumRuleCoordinator(BaseCoordinator):
    """Standard baseline coordinator: takes the minimum proposed current."""

    def coordinate(self, est: StateEstimate, task: Task,
                   cp: ChargingProposal, hp: HealthProposal,
                   tp: ThermalProposal) -> CoordinatorDecision:
        # Enforce minimum of charging proposal and health proposal
        i_cand = cp.current_a
        reason = "charging_speed_governs"
        if hp.max_c_rate is not None:
            i_health = hp.max_c_rate * (est.q_nom_ah * est.soh)
            if i_health < i_cand:
                i_cand = i_health
                reason = "health_cap_governs"
        return CoordinatorDecision(
            current_a=float(max(0.0, i_cand)),
            cool_cmd=float(tp.cool_cmd),
            coordinator_mode="minimum_rule",
            arbitration_reason=reason
        )


class DynamicPriorityCoordinator(BaseCoordinator):
    """State-dependent priority coordinator switching between speed, health, and thermal priorities."""

    def __init__(self, urgency_threshold_s: float = 600.0, temp_relax_c: float = 38.0):
        self.urgency_threshold_s = urgency_threshold_s
        self.temp_relax_c = temp_relax_c

    def coordinate(self, est: StateEstimate, task: Task,
                   cp: ChargingProposal, hp: HealthProposal,
                   tp: ThermalProposal) -> CoordinatorDecision:
        t_rem = max(task.deadline_s - est.time_s, 1.0)
        i_cand = cp.current_a
        i_health = hp.max_c_rate * (est.q_nom_ah * est.soh) if hp.max_c_rate is not None else cp.current_a
        i_req = hp.i_required_a if hp.i_required_a is not None else 0.0

        # Case 1: Deadline Urgency (At risk of failure)
        if hp.feasibility_status in ("AT_RISK", "INFEASIBLE") or t_rem < self.urgency_threshold_s:
            # Grant precedence to deadline completion up to physical health ceiling
            i_exec = max(i_req, min(i_cand, i_health * 1.15))
            reason = "deadline_urgency_precedence"
        # Case 2: Elevated Cell Temperature
        elif est.temp_meas > self.temp_relax_c or est.temp_pred > self.temp_relax_c:
            # Back off to minimum of health limit and 0.8x speed proposal
            i_exec = min(i_cand * 0.80, i_health)
            reason = "thermal_relaxation_precedence"
        # Case 3: Nominal Operation (Ample deadline margin)
        else:
            # Smooth compromise: weighted geometric mean between speed proposal and health ceiling
            i_exec = min(i_cand, i_health)
            reason = "nominal_health_mitigation"

        return CoordinatorDecision(
            current_a=float(max(0.0, i_exec)),
            cool_cmd=float(tp.cool_cmd),
            coordinator_mode="dynamic_priority",
            arbitration_reason=reason
        )


class ConstrainedQPCoordinator(BaseCoordinator):
    """Analytic convex quadratic programming coordinator.
    Minimizes:
        J(I) = w_speed * (I - I_speed)^2 + w_health * (I - I_health)^2 + w_thermal * (I^2 * R0)
    Subject to:
        I_min_req <= I <= I_max_limit
    """

    def __init__(self, base_w_speed: float = 1.0, base_w_health: float = 1.2, base_w_thermal: float = 0.5):
        self.w_s0 = base_w_speed
        self.w_h0 = base_w_health
        self.w_th0 = base_w_thermal

    def coordinate(self, est: StateEstimate, task: Task,
                   cp: ChargingProposal, hp: HealthProposal,
                   tp: ThermalProposal) -> CoordinatorDecision:
        t_rem = max(task.deadline_s - est.time_s, 1.0)
        soc_rem = max(task.target_soc - est.soc, 0.0)

        # Dynamic weight scheduling
        # As deadline approaches, speed weight scales up: w_s = w_s0 * (1 + (t_req / t_rem)^2)
        t_req = (soc_rem * 3600.0) / max(cp.current_a / max(est.q_nom_ah * est.soh, 0.5), 0.5)
        urgency_ratio = np.clip(t_req / t_rem, 0.2, 5.0)
        w_speed = self.w_s0 * float(urgency_ratio ** 1.5)

        # As resistance and degradation grow, health weight scales up
        w_health = self.w_h0 * float(max(est.r0_scale, 1.0)) * (1.1 - float(est.soh))

        # As temperature rises toward limit, thermal weight scales up exponentially
        temp_stress = float(np.clip((est.temp_meas - 25.0) / 15.0, 0.0, 2.5))
        w_thermal = self.w_th0 * (1.0 + float(temp_stress ** 2))

        # Proposals
        i_s = cp.current_a
        i_h = hp.max_c_rate * (est.q_nom_ah * est.soh) if hp.max_c_rate is not None else i_s
        r0 = 0.025 * max(est.r0_scale, 1.0)

        # Objective: J(I) = w_speed*(I - i_s)^2 + w_health*(I - i_h)^2 + w_thermal*r0*I^2
        # dJ/dI = 2*w_speed*(I - i_s) + 2*w_health*(I - i_h) + 2*w_thermal*r0*I = 0
        # (w_speed + w_health + w_thermal*r0) * I = w_speed * i_s + w_health * i_h
        denom = w_speed + w_health + w_thermal * r0
        if denom > 1e-6:
            i_opt = (w_speed * i_s + w_health * i_h) / denom
        else:
            i_opt = min(i_s, i_h)

        # Enforce bounds:
        # Lower bound: required current to meet deadline
        i_min = hp.i_required_a if (hp.i_required_a is not None and hp.feasibility_status == "AT_RISK") else 0.0
        # Upper bound: hardware limit & current charging proposal
        i_max = max(i_s, i_h)

        i_exec = float(np.clip(i_opt, i_min, i_max))

        return CoordinatorDecision(
            current_a=float(max(0.0, i_exec)),
            cool_cmd=float(tp.cool_cmd),
            coordinator_mode="constrained_qp",
            arbitration_reason="optimal_convex_compromise",
            weights={"w_speed": float(w_speed), "w_health": float(w_health), "w_thermal": float(w_thermal)}
        )
