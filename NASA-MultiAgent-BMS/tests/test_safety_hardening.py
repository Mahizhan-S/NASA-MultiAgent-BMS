"""Unit tests for Phase G Safety Supervisor Hardening.
Tests:
1. Rate-of-change temporal consistency check.
2. Stuck-sensor freeze detection under active current.
3. Thermal model energy-balance discrepancy detection.
4. Fault mitigation and conservative derating.
"""
from __future__ import annotations
import copy
import numpy as np
import pytest

from mabms.agents.base import StateEstimate
from mabms.agents.safety_agent import SafetyAgent
from mabms.config import load_config

cfg = load_config("configs/default.yaml")


def make_est(**kwargs):
    default = {
        "soc": 0.50,
        "soh": 1.0,
        "v1": 0.0,
        "v_meas": 3.80,
        "i_meas": 2.0,
        "temp_meas": 30.0,
        "temp_pred": 30.0,
        "time_s": 0.0,
        "q_nom_ah": 2.45,
        "r0_scale": 1.0,
    }
    default.update(kwargs)
    return StateEstimate(**default)


def test_safety_rate_of_change_violation_detection():
    c = copy.deepcopy(cfg)
    agent = SafetyAgent(c, enabled=True)
    
    # Step 1: Normal reading
    e1 = make_est(temp_meas=30.0, time_s=0.0)
    a1 = agent.filter(e1, 2.0, 0.0)
    assert not any("rate" in r for r in a1.reasons)

    # Step 2: Sudden unphysical jump of +5 °C in a single 10s step
    e2 = make_est(temp_meas=35.5, time_s=10.0)
    a2 = agent.filter(e2, 2.0, 0.0)
    assert any("rate_violation" in r for r in a2.reasons)
    assert agent.n_monitor_flags > 0


def test_safety_stuck_sensor_detection():
    c = copy.deepcopy(cfg)
    agent = SafetyAgent(c, enabled=True)

    # Run 7 steps with bitwise identical temperature under 2.5 A current
    stuck_flagged = False
    for step in range(7):
        e = make_est(temp_meas=30.0, time_s=step * 10.0, i_meas=2.5)
        act = agent.filter(e, 2.5, 0.0)
        if any("stuck" in r for r in act.reasons):
            stuck_flagged = True
            break

    assert stuck_flagged, "Safety agent should flag sensor freeze after multiple identical steps under load"


def test_safety_thermal_discrepancy_and_negative_bias():
    c = copy.deepcopy(cfg)
    c["safety"]["temp_monitor"]["thr_c"] = 3.0
    agent = SafetyAgent(c, enabled=True)

    # Inject active 4.0 A charging current while sensor is biased low at 30.0 °C
    bias_flagged = False
    for step in range(15):
        e = make_est(temp_meas=30.0, time_s=step * 10.0, i_meas=4.0)
        act = agent.filter(e, 4.0, 0.0)
        if any("bias" in r or "implausible" in r for r in act.reasons):
            bias_flagged = True
            break

    assert bias_flagged or agent.n_monitor_flags > 0


def test_safety_derates_current_when_elevated_temperature():
    c = copy.deepcopy(cfg)
    agent = SafetyAgent(c, enabled=True)

    # Hot cell at 44.5 °C should be throttled drastically
    e_hot = make_est(temp_meas=44.5, temp_pred=44.8)
    act = agent.filter(e_hot, 4.0, 0.0)
    assert act.current_a < 1.0
    assert act.cool_cmd == 1.0
    assert any("thermal_derate" in r for r in act.reasons)
