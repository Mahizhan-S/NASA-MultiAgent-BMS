"""Unit tests for Phase H Multi-Agent Coordinators.
Tests:
1. MinimumRuleCoordinator arbitration.
2. DynamicPriorityCoordinator urgency and thermal state switching.
3. ConstrainedQPCoordinator analytic optimization.
4. End-to-end execution in simulation episode.
"""
from __future__ import annotations
import copy
import pytest

from mabms.agents.base import ChargingProposal, HealthProposal, StateEstimate, Task, ThermalProposal
from mabms.agents.coordinator import (
    MinimumRuleCoordinator,
    DynamicPriorityCoordinator,
    ConstrainedQPCoordinator,
)
from mabms.config import load_config
from mabms.environment.battery_env import Scenario
from mabms.evaluation.run_independent import run_episode

cfg = load_config("configs/default.yaml")


def make_est(**kwargs):
    d = {
        "soc": 0.50,
        "soh": 0.90,
        "v1": 0.0,
        "v_meas": 3.80,
        "i_meas": 2.0,
        "temp_meas": 30.0,
        "temp_pred": 30.0,
        "time_s": 500.0,
        "q_nom_ah": 2.45,
        "r0_scale": 1.1,
    }
    d.update(kwargs)
    return StateEstimate(**d)


def test_minimum_rule_coordinator():
    coord = MinimumRuleCoordinator()
    est = make_est()
    task = Task(0.9, 3600.0)
    cp = ChargingProposal(current_a=3.0, mode="CC")
    # Health cap allows 1.0 C = 1.0 * 2.45 * 0.9 = 2.205 A
    hp = HealthProposal(max_c_rate=1.0)
    tp = ThermalProposal(cool_cmd=0.5)

    dec = coord.coordinate(est, task, cp, hp, tp)
    assert dec.current_a == pytest.approx(2.205, rel=1e-3)
    assert dec.cool_cmd == 0.5
    assert dec.arbitration_reason == "health_cap_governs"


def test_dynamic_priority_coordinator_urgency():
    coord = DynamicPriorityCoordinator(urgency_threshold_s=600.0)
    est = make_est(time_s=3100.0) # 500s remaining < 600s urgency threshold
    task = Task(0.9, 3600.0)
    cp = ChargingProposal(current_a=3.5, mode="CC")
    hp = HealthProposal(max_c_rate=1.0, feasibility_status="AT_RISK", i_required_a=2.6)
    tp = ThermalProposal(cool_cmd=1.0)

    dec = coord.coordinate(est, task, cp, hp, tp)
    assert dec.arbitration_reason == "deadline_urgency_precedence"
    # Should give priority to required current
    assert dec.current_a >= 2.6


def test_constrained_qp_coordinator():
    coord = ConstrainedQPCoordinator()
    est = make_est()
    task = Task(0.9, 3600.0)
    cp = ChargingProposal(current_a=3.0, mode="CC")
    hp = HealthProposal(max_c_rate=1.0, feasibility_status="FEASIBLE", i_required_a=1.5)
    tp = ThermalProposal(cool_cmd=0.2)

    dec = coord.coordinate(est, task, cp, hp, tp)
    assert 1.5 <= dec.current_a <= 3.0
    assert dec.coordinator_mode == "constrained_qp"
    assert "w_speed" in dec.weights


def test_coordinator_in_closed_loop_episode():
    sc = Scenario(soc0=0.3, target_soc=0.8, deadline_s=2500.0, soh0=0.85)
    coord = ConstrainedQPCoordinator()
    r = run_episode(
        cfg,
        sc,
        method="independent_adaptive_health",
        seed=42,
        options={"coordinator": coord}
    )
    assert r["steps"] > 0
    assert r["target_reached"]
    assert r["unsafe_executed_steps"] == 0
