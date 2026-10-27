"""Dedicated Acceptance Test Suite for the Charging Agent (Phase I Hardening).
Validates all requirements and edge cases identified in the audit:
1. Negative max_c_rate rejection & non-negative proposal guarantee.
2. Invalid StateEstimate (is_valid=False) rejection in ChargingAgent and SafetyAgent.
3. Invalid parameter rejection (negative capacity, target_soc > 1.0, negative deadline, NaN v_cv).
4. Deadline calculation with coulombic efficiency and bound clamping.
5. CC-to-CV transition behavior and voltage limiting.
6. Stop latch and hysteresis behavior.
7. Explicit reset() across independent episodes.
8. Closed-loop paired baseline integration.
"""
from __future__ import annotations
import copy
import numpy as np
import pytest

from mabms.config import load_config
from mabms.agents.base import StateEstimate, Task, ChargingProposal, SafeAction
from mabms.agents.charging_agent import ChargingAgent
from mabms.agents.safety_agent import SafetyAgent
from mabms.environment.battery_env import Scenario
from mabms.evaluation.run_independent import run_episode

CFG = load_config("configs/default.yaml")


# ==============================================================================
# Issue 1: Non-Negative Current Proposal Guarantee
# ==============================================================================

def test_negative_max_c_rate_rejected_and_non_negative_proposal():
    """Issue 1: Caller-provided negative max_c_rate must not produce negative current proposals."""
    ca = ChargingAgent(CFG)
    est = StateEstimate(soc=0.40, soh=1.0, v1=0.0, v_meas=3.80, i_meas=1.0,
                        temp_meas=25.0, temp_pred=25.0, time_s=100.0, q_nom_ah=2.45)
    task = Task(target_soc=0.85, deadline_s=3600.0)

    # Test negative max_c_rate values
    for bad_rate in [-1.0, -0.01, -10.0]:
        prop = ca.propose(est, task, max_c_rate=bad_rate)
        # Must be rejected as invalid or clamped to 0.0, never negative
        assert prop.current_a >= 0.0, f"Negative current proposed: {prop.current_a}"
        assert prop.current_a == 0.0, f"Expected 0.0 current for negative limit, got {prop.current_a}"

    # Normal positive max_c_rate must work
    prop_valid = ca.propose(est, task, max_c_rate=1.0)
    assert prop_valid.current_a > 0.0
    assert prop_valid.current_a <= 1.0 * est.q_nom_ah + 1e-6


# ==============================================================================
# Issue 2: Cross-Agent Invalid-State Regression (is_valid=False)
# ==============================================================================

def test_invalid_state_estimate_is_valid_false_triggers_fallback_in_charger_and_safety():
    """Issue 2: When a finite StateEstimate has is_valid=False, both ChargingAgent
    and SafetyAgent must reject the state and enforce conservative fallback (0.0 A)."""
    ca = ChargingAgent(CFG)
    sa = SafetyAgent(CFG)
    task = Task(target_soc=0.85, deadline_s=3600.0)

    # Finite, numerically plausible state but explicitly flagged invalid (e.g. from upstream sensor fault)
    est_invalid = StateEstimate(
        soc=0.50, soh=0.90, v1=0.0, v_meas=3.85, i_meas=1.0,
        temp_meas=26.0, temp_pred=26.0, time_s=120.0, q_nom_ah=2.45,
        is_valid=False
    )

    # 1. ChargingAgent must reject and return safe 0.0 A fallback
    prop = ca.propose(est_invalid, task)
    assert prop.current_a == 0.0, f"ChargingAgent proposed positive current on invalid state: {prop.current_a}"
    assert prop.is_fallback is True
    assert prop.is_valid is False
    assert "fallback" in prop.mode

    # 2. SafetyAgent must also independently catch is_valid=False and enforce fallback
    # Even if an unhardened proposer proposed 2.0 A, SafetyAgent must intervene
    act = sa.filter(est_invalid, i_prop=2.0, cool_prop=0.0)
    assert act.current_a == 0.0, f"SafetyAgent permitted charging on invalid state: {act.current_a}"
    assert act.fallback is True
    assert act.intervened is True
    assert "invalid_state_or_action" in act.reasons


# ==============================================================================
# Issue 3: Comprehensive Parameter Validation
# ==============================================================================

def test_invalid_parameters_rejected():
    """Issue 3: Invalid state, capacity, target SOC, deadline, and CV voltages must be rejected."""
    ca = ChargingAgent(CFG)
    task_nom = Task(target_soc=0.85, deadline_s=3600.0)
    est_nom = StateEstimate(soc=0.40, soh=1.0, v1=0.0, v_meas=3.80, i_meas=1.0,
                            temp_meas=25.0, temp_pred=25.0, time_s=100.0, q_nom_ah=2.45)

    # (a) Negative nominal capacity
    est_neg_q = copy.deepcopy(est_nom)
    est_neg_q.q_nom_ah = -2.45
    p_neg_q = ca.propose(est_neg_q, task_nom)
    assert p_neg_q.current_a == 0.0
    assert p_neg_q.is_fallback is True

    # (b) Target SOC > 1.0 (e.g. 1.5) or <= 0.0
    for bad_soc in [1.5, 2.0, 0.0, -0.1]:
        task_bad_soc = Task(target_soc=bad_soc, deadline_s=3600.0)
        p_bad_soc = ca.propose(est_nom, task_bad_soc)
        assert p_bad_soc.current_a == 0.0
        assert p_bad_soc.is_fallback is True

    # (c) Negative or zero deadline
    for bad_dl in [0.0, -100.0, float("nan"), float("inf")]:
        task_bad_dl = Task(target_soc=0.85, deadline_s=bad_dl)
        p_bad_dl = ca.propose(est_nom, task_bad_dl)
        assert p_bad_dl.current_a == 0.0
        assert p_bad_dl.is_fallback is True

    # (d) Malformed v_cv (NaN or outside realistic range)
    for bad_vcv in [float("nan"), 2.0, 5.0]:
        p_bad_vcv = ca.propose(est_nom, task_nom, v_cv=bad_vcv)
        assert p_bad_vcv.current_a == 0.0
        assert p_bad_vcv.is_fallback is True

    # (e) Negative timestamp
    est_neg_time = copy.deepcopy(est_nom)
    est_neg_time.time_s = -10.0
    p_neg_time = ca.propose(est_neg_time, task_nom)
    assert p_neg_time.current_a == 0.0
    assert p_neg_time.is_fallback is True


# ==============================================================================
# Issue 4: Deadline Calculation with Coulombic Efficiency
# ==============================================================================

def test_deadline_calculation_with_coulombic_efficiency_and_limits():
    """Issue 4: Required C-rate must account for configured coulombic efficiency and bound limits."""
    ca = ChargingAgent(CFG)
    eta = CFG["simulator"]["coulombic_eff"]  # 0.995

    # Case A: Nominal deadline demand
    soc_start = 0.30
    soc_target = 0.90
    delta_soc = soc_target - soc_start  # 0.60
    time_left = 1800.0  # 30 minutes
    est = StateEstimate(soc=soc_start, soh=1.0, v1=0.0, v_meas=3.65, i_meas=1.0,
                        temp_meas=25.0, temp_pred=25.0, time_s=0.0, q_nom_ah=2.45)
    task = Task(target_soc=soc_target, deadline_s=time_left)

    prop = ca.propose(est, task)

    # Theoretical required C-rate with coulombic efficiency
    expected_req_c = (delta_soc / eta) * (3600.0 / time_left)
    assert abs(prop.required_c - expected_req_c) < 1e-4

    # Proposed C-rate should apply deadline margin
    expected_c = min(expected_req_c * ca.margin, CFG["limits"]["i_max_c"])
    expected_i_cc = expected_c * est.q_nom_ah
    assert abs(prop.current_a - expected_i_cc) < 1e-3

    # Case B: Target already reached (delta_soc <= 0)
    est_done = StateEstimate(soc=0.92, soh=1.0, v1=0.0, v_meas=4.15, i_meas=0.0,
                             temp_meas=25.0, temp_pred=25.0, time_s=100.0, q_nom_ah=2.45)
    prop_done = ca.propose(est_done, task)
    assert prop_done.current_a == 0.0
    assert prop_done.mode == "done"
    assert prop_done.required_c == 0.0

    # Case C: Impossible deadline (50% SOC in 10 seconds -> ~180 C demand)
    task_impossible = Task(target_soc=0.90, deadline_s=10.0)
    prop_imp = ca.propose(est, task_impossible)
    # Must be saturated at absolute hardware current limit
    assert prop_imp.current_a <= ca.i_abs + 1e-6
    assert prop_imp.current_a <= CFG["limits"]["i_max_c"] * est.q_nom_ah + 1e-6


# ==============================================================================
# Issue 5: CC-to-CV Transitions and Voltage Limiting
# ==============================================================================

def test_cc_to_cv_transition_and_voltage_limiting():
    """Verify smooth CC to CV transition as battery approaches terminal voltage limit."""
    ca = ChargingAgent(CFG)
    task = Task(target_soc=0.95, deadline_s=7200.0)  # relaxed deadline -> CC governed by nominal rate

    # 1. Low SOC: terminal voltage comfortably below V_max -> CC governs
    est_cc = StateEstimate(soc=0.30, soh=1.0, v1=0.0, v_meas=3.65, i_meas=1.0,
                           temp_meas=25.0, temp_pred=25.0, time_s=100.0, q_nom_ah=2.45)
    prop_cc = ca.propose(est_cc, task)
    assert prop_cc.mode == "CC"
    assert prop_cc.current_a > 1.0

    # 2. High SOC: terminal voltage near V_max -> CV governs
    # At SOC=0.94, OCV is ~4.14 V, allowing limited current before reaching 4.20 V
    est_cv = StateEstimate(soc=0.94, soh=1.0, v1=0.0, v_meas=4.16, i_meas=1.0,
                           temp_meas=25.0, temp_pred=25.0, time_s=2500.0, q_nom_ah=2.45)
    prop_cv = ca.propose(est_cv, task)
    assert prop_cv.mode == "CV"
    assert prop_cv.current_a < prop_cc.current_a

    # 3. Alternate v_cv input (e.g. Health Agent gentle CV ceiling: 4.15 V vs 4.20 V)
    prop_vcv_low = ca.propose(est_cv, task, v_cv=4.15)
    assert prop_vcv_low.current_a <= prop_cv.current_a


# ==============================================================================
# Issue 6: Stop Latch & Hysteresis Correctness
# ==============================================================================

def test_stop_latch_and_hysteresis_behavior():
    """Verify that reaching target SOC latches stop and hysteresis prevents chatter."""
    cfg_hyst = copy.deepcopy(CFG)
    cfg_hyst["agents"]["charging"]["restart_hysteresis"] = 0.02
    ca = ChargingAgent(cfg_hyst)
    task = Task(target_soc=0.90, deadline_s=3600.0)

    # Charge below target
    est_before = StateEstimate(soc=0.88, soh=1.0, v1=0.0, v_meas=4.05, i_meas=1.0,
                               temp_meas=25.0, temp_pred=25.0, time_s=1000.0, q_nom_ah=2.45)
    assert ca.propose(est_before, task).current_a > 0.0

    # Reach target
    est_reach = StateEstimate(soc=0.905, soh=1.0, v1=0.0, v_meas=4.18, i_meas=1.0,
                              temp_meas=25.0, temp_pred=25.0, time_s=1500.0, q_nom_ah=2.45)
    prop_reach = ca.propose(est_reach, task)
    assert prop_reach.current_a == 0.0
    assert prop_reach.mode == "done"
    assert ca._stopped is True

    # Slight drop within hysteresis margin (0.898 >= 0.90 - 0.02) -> latch holds (no current)
    est_dip = StateEstimate(soc=0.898, soh=1.0, v1=0.0, v_meas=4.10, i_meas=0.0,
                            temp_meas=25.0, temp_pred=25.0, time_s=1520.0, q_nom_ah=2.45)
    prop_dip = ca.propose(est_dip, task)
    assert prop_dip.current_a == 0.0
    assert prop_dip.mode == "done"

    # Drop clearly below hysteresis threshold (0.86 < 0.90 - 0.02) -> resumes charging
    est_resume = StateEstimate(soc=0.86, soh=1.0, v1=0.0, v_meas=4.00, i_meas=0.0,
                               temp_meas=25.0, temp_pred=25.0, time_s=1550.0, q_nom_ah=2.45)
    prop_resume = ca.propose(est_resume, task)
    assert prop_resume.current_a > 0.0
    assert ca._stopped is False


# ==============================================================================
# Issue 7: Explicit Reset Lifecycle Across Independent Episodes
# ==============================================================================

def test_charging_agent_explicit_reset_across_episodes():
    """Verify that reset() cleans up stop latch so an instance can be safely reused."""
    ca = ChargingAgent(CFG)
    task1 = Task(target_soc=0.85, deadline_s=3600.0)

    # Episode 1 completes
    est_end_ep1 = StateEstimate(soc=0.86, soh=1.0, v1=0.0, v_meas=4.12, i_meas=0.5,
                                temp_meas=25.0, temp_pred=25.0, time_s=2000.0, q_nom_ah=2.45)
    prop1 = ca.propose(est_end_ep1, task1)
    assert prop1.current_a == 0.0
    assert ca._stopped is True

    # Start Episode 2 without reset -> would be erroneously latched if reset is not called
    ca.reset()
    assert ca._stopped is False

    task2 = Task(target_soc=0.90, deadline_s=4000.0)
    est_start_ep2 = StateEstimate(soc=0.20, soh=1.0, v1=0.0, v_meas=3.60, i_meas=0.0,
                                  temp_meas=25.0, temp_pred=25.0, time_s=0.0, q_nom_ah=2.45)
    prop2 = ca.propose(est_start_ep2, task2)
    assert prop2.current_a > 0.0
    assert prop2.mode == "CC"


# ==============================================================================
# Issue 8: Closed-Loop Integration Validation
# ==============================================================================

def test_closed_loop_paired_baselines():
    """Verify that the hardened ChargingAgent operates correctly in closed-loop simulation."""
    sc = Scenario(soc0=0.25, target_soc=0.85, deadline_s=3600.0, soh0=0.85)

    # 1. Conventional baseline run
    r_conv = run_episode(CFG, sc, method="conventional", seed=42, safety_enabled=True)
    assert r_conv["target_reached"] is True
    assert r_conv["unsafe_executed_steps"] == 0

    # 2. Independent deadline-aware run
    r_indep = run_episode(CFG, sc, method="independent", seed=42, safety_enabled=True)
    assert r_indep["target_reached"] is True
    assert r_indep["unsafe_executed_steps"] == 0

    # 3. Adaptive Health Agent integration run
    r_adapt = run_episode(CFG, sc, method="independent_adaptive_health", seed=42, safety_enabled=True)
    assert r_adapt["target_reached"] is True
    assert r_adapt["unsafe_executed_steps"] == 0
    assert r_adapt["capacity_loss"] < r_conv["capacity_loss"]
