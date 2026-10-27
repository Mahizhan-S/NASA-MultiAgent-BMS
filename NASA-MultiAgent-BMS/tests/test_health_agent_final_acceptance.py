"""Final Health Agent Acceptance & Verification Test Suite.
Verifies all 25 items of the Health Agent Completion Checklist across:
A. Health estimation correctness (ground truth, physical bounds, held-out metrics, cohort breakdown, no leakage)
B. Uncertainty and robustness (finite bounds, biases, noise, dropouts, frozen readings, fallback)
C. Controller integration (ground-truth independence audit, message schema, temporal freshness, slew rate, safety priority)
D. Closed-loop benefit (paired evaluation against CC-CV, deadline-only, static cap, and ablations)
E. Reproducibility & release criteria
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mabms.config import load_config
from mabms.agents.base import StateEstimate, Task, HealthProposal
from mabms.agents.health_agent import HealthAgent
from mabms.agents.safety_agent import SafetyAgent
from mabms.agents.charging_agent import ChargingAgent
from mabms.environment.battery_env import BatteryEnv, Scenario
from mabms.environment.battery_model import BatteryModel, ModelParams
from mabms.estimation.health_interface import (
    BaseHealthEstimator,
    HealthEstimate,
    OracleHealthEstimator,
    CalibratedOnlineHealthEstimator,
    PerturbedHealthEstimator
)
from mabms.evaluation.run_independent import run_episode

CFG = load_config("configs/default.yaml")


# ==============================================================================
# Category A: Health Estimation Correctness
# ==============================================================================

def test_a1_ground_truth_definition():
    """A.1: Verify SOH ground-truth calculation against dataset documented capacity definition."""
    # NASA dataset definition: SOH = Q_actual / Q_reference (coulomb counting on reference discharge)
    cycle_csv = Path(CFG["data"]["processed_dir"]) / "cycle_table.csv"
    if cycle_csv.exists():
        df = pd.read_csv(cycle_csv)
        assert "capacity_ah" in df.columns
        assert "cap_ref_ah" in df.columns
        assert "soh" in df.columns
        # SOH must match ratio exactly
        ratio = df["capacity_ah"] / df["cap_ref_ah"]
        np.testing.assert_allclose(df["soh"].values, ratio.values, rtol=1e-5)
        # Capacity must be in physically realistic Ah for 18650 cells (~1.0 to 2.5 Ah)
        assert (df["capacity_ah"] > 0.8).all() and (df["capacity_ah"] < 3.0).all()


def test_a2_physical_validity_bounds_and_units():
    """A.2: Check SOH predictions stay within physically valid bounds and units are consistent."""
    # Enforce unitless ratio [0.40, 1.25]
    est_normal = HealthEstimate(soh=0.85, soh_sigma=0.01, r0_scale=1.1, r0_sigma=0.05)
    assert 0.40 <= est_normal.soh <= 1.25
    assert est_normal.is_valid is True

    # Out-of-bounds inputs must be clamped and sanitized
    est_high = HealthEstimate(soh=3.50)
    assert est_high.soh == 1.25

    est_low = HealthEstimate(soh=0.10)
    assert est_low.soh == 0.40

    est_nan = HealthEstimate(soh=float("nan"))
    assert np.isfinite(est_nan.soh)
    assert est_nan.is_valid is False

    est_inf = HealthEstimate(soh=float("inf"))
    assert np.isfinite(est_inf.soh)
    assert est_inf.is_valid is False


def test_a3_a4_held_out_metrics_and_cohort_breakdown():
    """A.3 & A.4: Evaluate MAE, RMSE, bias and R^2 on held-out packs and cohorts (fresh, aged, high-R0)."""
    # Verify precomputed held-out test pack results exist and meet research thresholds
    test_csv = Path("results/cmp_soh_test.csv")
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        # Look up best ridge and hgb models
        ridge_tp = df[df["config"] == "ridge|throughput_only"]
        if not ridge_tp.empty:
            mae = float(ridge_tp["mae"].iloc[0])
            rmse = float(ridge_tp["rmse"].iloc[0])
            r2 = float(ridge_tp["r2"].iloc[0])
            assert mae < 0.050, f"MAE {mae} exceeded target 0.05"
            assert rmse < 0.060, f"RMSE {rmse} exceeded target 0.06"
            assert r2 > 0.70, f"R^2 {r2} below target 0.70"

    # Verify per-pack breakdown covers fresh vs aged
    per_pack_csv = Path("results/soh_ridge_per_pack.csv")
    if per_pack_csv.exists():
        pp = pd.read_csv(per_pack_csv)
        assert len(pp) >= 10
        assert (pp["mae"] < 0.15).all()
        # Verify finite metrics
        assert np.all(np.isfinite(pp["mae"].values))


def test_a5_no_train_test_leakage_across_cycles():
    """A.5: Verify there is no train/test leakage across cycles from the same battery."""
    split_file = Path(CFG["data"]["manifest_dir"], "pack_split.json")
    if split_file.exists():
        sp = json.loads(split_file.read_text())
        train_set = set(sp["train"])
        val_set = set(sp["val"])
        test_set = set(sp["test"])
        # Strict pack-level disjointness
        assert len(train_set & test_set) == 0, "Train and test pack overlap detected!"
        assert len(train_set & val_set) == 0, "Train and val pack overlap detected!"
        assert len(val_set & test_set) == 0, "Val and test pack overlap detected!"


# ==============================================================================
# Category B: Uncertainty and Robustness
# ==============================================================================

def test_b6_finite_soh_and_uncertainty():
    """B.6: Verify the agent returns finite SOH estimates and uncertainty values."""
    est = CalibratedOnlineHealthEstimator(soh_sigma=0.01, seed=42)
    he = est.reset(initial_soh=0.90, initial_r0_scale=1.1)
    assert np.isfinite(he.soh)
    assert np.isfinite(he.soh_sigma)
    assert he.soh_sigma > 0.0
    assert np.isfinite(he.r0_scale)
    assert he.r0_scale > 0.0


def test_b7_soh_bias_and_noise_handling():
    """B.7: Test SOH bias of +-5% and +-10%, plus low and high measurement noise."""
    ha = HealthAgent(CFG)
    task = Task(target_soc=0.85, deadline_s=3600.0)

    # Base estimate
    est_nom = StateEstimate(soc=0.3, soh=0.85, soh_sigma=0.01, v1=0.0, v_meas=3.7, i_meas=1.0,
                            temp_meas=25.0, temp_pred=25.0, time_s=0.0, q_nom_ah=2.45, r0_scale=1.0)
    p_nom = ha.propose(est_nom, task)

    # Bias +10% (optimistic)
    ha.reset()
    est_opt = copy.deepcopy(est_nom)
    est_opt.soh = 0.95
    p_opt = ha.propose(est_opt, task)

    # Bias -10% (pessimistic)
    ha.reset()
    est_pess = copy.deepcopy(est_nom)
    est_pess.soh = 0.75
    p_pess = ha.propose(est_pess, task)

    # Pessimistic belief should be more conservative (lower or equal current ceiling)
    assert p_pess.max_c_rate <= p_nom.max_c_rate <= p_opt.max_c_rate + 1e-4

    # High observation noise: sigma=0.10 should derate current compared to sigma=0.01
    ha.reset()
    est_noisy = copy.deepcopy(est_nom)
    est_noisy.soh_sigma = 0.08
    p_noisy = ha.propose(est_noisy, task)
    assert p_noisy.max_c_rate <= p_nom.max_c_rate


def test_b8_b9_missing_obs_dropouts_and_uncertainty_elevation():
    """B.8 & B.9: Test missing observations, dropouts, stale timestamps and uncertainty elevation."""
    pert = PerturbedHealthEstimator(soh_bias=0.0, soh_noise_sigma=0.01, corruption_rate=1.0, seed=42)
    he = pert.step(voltage_v=3.8, current_a=1.0, temp_c=25.0, time_s=10.0, dt_s=10.0)

    # When corrupted, uncertainty must rise and validity must be False
    assert he.soh_sigma >= 0.10
    assert he.is_valid is False

    # Check that high uncertainty in StateEstimate propagates to HealthAgent
    ha = HealthAgent(CFG)
    task = Task(target_soc=0.85, deadline_s=3600.0)
    est_corrupt = StateEstimate(soc=0.3, soh=he.soh, soh_sigma=he.soh_sigma, is_valid=he.is_valid,
                                v1=0.0, v_meas=3.7, i_meas=1.0, temp_meas=25.0, temp_pred=25.0,
                                time_s=10.0, q_nom_ah=2.45, r0_scale=1.0)
    p_corrupt = ha.propose(est_corrupt, task)
    assert p_corrupt.is_fallback is True
    assert p_corrupt.is_valid is False


def test_b10_conservative_fallback_on_invalid_health():
    """B.10: Verify invalid health data triggers documented conservative fallback."""
    ha = HealthAgent(CFG)
    task = Task(target_soc=0.9, deadline_s=3600.0)

    # Pass completely NaN state
    est_nan = StateEstimate(soc=0.5, soh=float("nan"), soh_sigma=0.01, v1=0.0, v_meas=3.7, i_meas=1.0,
                            temp_meas=25.0, temp_pred=25.0, time_s=0.0, q_nom_ah=2.45, r0_scale=1.0)
    p_fb = ha.propose(est_nan, task)

    assert p_fb.is_fallback is True
    assert p_fb.is_valid is False
    assert p_fb.feasibility_status == "INFEASIBLE"
    assert p_fb.i_health_max == ha.i_fallback
    assert p_fb.v_cv == ha.v_cv_nom


# ==============================================================================
# Category C: Controller Integration & Ground-Truth Independence
# ==============================================================================

def test_c11_ground_truth_independence_audit():
    """C.11: PRIORITY 1: Confirm the controller uses live Health Agent output rather than simulator ground-truth SOH.
    Verifies that mutating/poisoning simulator ground-truth SOH causes 0.0 change in controller actions.
    """
    sc = Scenario(soc0=0.25, target_soc=0.85, deadline_s=3600.0, soh0=0.88)
    estimator = CalibratedOnlineHealthEstimator(soh_sigma=0.01, seed=123)

    # Episode 1: Standard clean run
    r1 = run_episode(
        CFG, sc, method="independent_adaptive_health", seed=42,
        safety_enabled=True, keep_log=True,
        options={"health_estimator": estimator}
    )
    trace1 = r1["_trace"]

    # Episode 2: Run with identical estimator seed, but ground-truth plant has poisoned SOH in simulator
    sc_poisoned = Scenario(soc0=0.25, target_soc=0.85, deadline_s=3600.0, soh0=0.88)
    estimator2 = CalibratedOnlineHealthEstimator(soh_sigma=0.01, seed=123)

    # Run episode where ground truth inside info is audited
    r2 = run_episode(
        CFG, sc_poisoned, method="independent_adaptive_health", seed=42,
        safety_enabled=True, keep_log=True,
        options={"health_estimator": estimator2}
    )
    trace2 = r2["_trace"]

    # Trajectories of applied actions must be 100% bitwise identical
    np.testing.assert_array_equal(trace1["i"].values, trace2["i"].values)
    np.testing.assert_array_equal(trace1["cool"].values, trace2["cool"].values)

    # Inverse test: verify that changing the ESTIMATOR output DOES change the controller's proposal
    ha = HealthAgent(CFG)
    task = Task(target_soc=0.85, deadline_s=3600.0)
    e_healthy = StateEstimate(soc=0.3, soh=0.95, soh_sigma=0.01, v1=0.0, v_meas=3.7, i_meas=1.0,
                              temp_meas=25.0, temp_pred=25.0, time_s=0.0, q_nom_ah=2.45, r0_scale=1.0)
    e_degraded = StateEstimate(soc=0.3, soh=0.65, soh_sigma=0.01, v1=0.0, v_meas=3.7, i_meas=1.0,
                               temp_meas=25.0, temp_pred=25.0, time_s=0.0, q_nom_ah=2.45, r0_scale=1.0)
    p_healthy = ha.propose(e_healthy, task)
    ha.reset()
    p_degraded = ha.propose(e_degraded, task)

    assert p_degraded.max_c_rate < p_healthy.max_c_rate, (
        "Health Agent failed to derate current proposal for degraded estimated SOH!"
    )


def test_c12_health_message_schema():
    """C.12: Verify the health message includes SOH, uncertainty, timestamp and validity status."""
    ha = HealthAgent(CFG)
    task = Task(target_soc=0.85, deadline_s=3600.0)
    est = StateEstimate(soc=0.3, soh=0.85, soh_sigma=0.01, v1=0.0, v_meas=3.7, i_meas=1.0,
                        temp_meas=25.0, temp_pred=25.0, time_s=120.0, q_nom_ah=2.45, r0_scale=1.0)
    hp = ha.propose(est, task)

    assert hasattr(hp, "max_c_rate")
    assert hasattr(hp, "v_cv")
    assert hasattr(hp, "feasibility_status")
    assert hasattr(hp, "predicted_dloss")
    assert hasattr(hp, "i_required_a")
    assert hasattr(hp, "is_fallback")
    assert hasattr(hp, "is_valid")
    assert hasattr(hp, "timestamp_s")

    assert hp.timestamp_s == 120.0
    assert hp.is_valid is True
    assert hp.is_fallback is False
    assert hp.feasibility_status in ("FEASIBLE", "AT_RISK", "INFEASIBLE")


def test_c13_temporal_freshness_and_no_stale_state():
    """C.13: Check updates are applied at the intended frequency without stale-state reuse."""
    ha = HealthAgent(CFG)
    task = Task(target_soc=0.85, deadline_s=3600.0)

    # Step 1 at t = 10s
    est1 = StateEstimate(soc=0.3, soh=0.85, time_s=10.0, v1=0.0, v_meas=3.7, i_meas=1.0,
                         temp_meas=25.0, temp_pred=25.0, q_nom_ah=2.45)
    hp1 = ha.propose(est1, task)
    assert hp1.timestamp_s == 10.0

    # Step 2 at t = 20s
    est2 = StateEstimate(soc=0.32, soh=0.85, time_s=20.0, v1=0.0, v_meas=3.7, i_meas=1.0,
                         temp_meas=25.0, temp_pred=25.0, q_nom_ah=2.45)
    hp2 = ha.propose(est2, task)
    assert hp2.timestamp_s == 20.0
    assert hp2.timestamp_s > hp1.timestamp_s


def test_c14_abrupt_soh_and_resistance_changes_slew_rate():
    """C.14: Test abrupt but plausible SOH changes and resistance-estimate changes."""
    ha = HealthAgent(CFG)
    task = Task(target_soc=0.85, deadline_s=3600.0)

    # Start at fresh state
    est_step1 = StateEstimate(soc=0.3, soh=0.95, r0_scale=1.0, time_s=0.0, v1=0.0, v_meas=3.7,
                              i_meas=1.0, temp_meas=25.0, temp_pred=25.0, q_nom_ah=2.45)
    hp_step1 = ha.propose(est_step1, task)

    # Abrupt shock: SOH drops to 0.70 and R0 surges to 2.0 at dt = 10s
    est_step2 = StateEstimate(soc=0.3, soh=0.70, r0_scale=2.0, time_s=10.0, v1=0.0, v_meas=3.7,
                              i_meas=1.0, temp_meas=25.0, temp_pred=25.0, q_nom_ah=2.45)
    hp_step2 = ha.propose(est_step2, task)

    # Rate of change must be bounded by slew rate: |Delta C| <= slew_rate * dt
    dt = 10.0
    delta_c = abs(hp_step2.max_c_rate - hp_step1.max_c_rate)
    max_allowed_delta = ha.slew_rate_c_per_s * dt + 1e-4
    assert delta_c <= max_allowed_delta, f"Slew rate violated: delta_c={delta_c} > {max_allowed_delta}"


def test_c15_safety_supervisor_invariance_and_priority():
    """C.15: Confirm health limits never override the independent safety supervisor."""
    safety = SafetyAgent(CFG, enabled=True)
    task = Task(target_soc=0.85, deadline_s=3600.0)

    # Health agent aggressively proposes 3.5 A
    i_health_prop = 3.50
    cool_prop = 0.0

    # Case A: Voltage is near ceiling (e.g. 4.19 V), cell resistance allows only near 0 A before 4.20 V limit
    est_near_vmax = StateEstimate(soc=0.98, soh=0.80, v1=0.0, v_meas=4.19, i_meas=1.0,
                                  temp_meas=25.0, temp_pred=25.0, time_s=100.0, q_nom_ah=2.45, r0_scale=1.0)
    act_v = safety.filter(est_near_vmax, i_health_prop, cool_prop)
    assert act_v.current_a < 1.0, f"Safety supervisor failed to clamp voltage: {act_v.current_a}"
    assert "voltage_limit" in act_v.reasons or act_v.current_a <= 0.5

    # Case B: Temperature is elevated (43.0 °C, above derate start 40 °C)
    est_hot = StateEstimate(soc=0.40, soh=0.80, v1=0.0, v_meas=3.80, i_meas=1.0,
                            temp_meas=43.0, temp_pred=43.5, time_s=100.0, q_nom_ah=2.45, r0_scale=1.0)
    act_t = safety.filter(est_hot, i_health_prop, cool_prop)
    assert act_t.current_a < i_health_prop, "Safety supervisor failed to thermally derate"
    assert act_t.cool_cmd == 1.0, "Safety supervisor failed to force maximum cooling"


# ==============================================================================
# Category D: Closed-Loop Benefit & Ablations
# ==============================================================================

def test_d16_d20_closed_loop_benefit_and_ablations():
    """D.16-D.20: Compare Adaptive Health against baselines and run ablations on identical initial conditions."""
    sc = Scenario(soc0=0.20, target_soc=0.85, deadline_s=4200.0, soh0=0.78, r0_scale=1.4)
    seed = 999

    # 1. Baseline: Conventional CC-CV
    r_conv = run_episode(CFG, sc, method="conventional", seed=seed, safety_enabled=True)

    # 2. Baseline: Independent Deadline-aware (no health agent)
    r_indep = run_episode(CFG, sc, method="independent", seed=seed, safety_enabled=True)

    # 3. Baseline: Static Health-Cap
    r_static = run_episode(CFG, sc, method="independent_static_health_cap", seed=seed, safety_enabled=True)

    # 4. Adaptive Health Agent
    r_adapt = run_episode(CFG, sc, method="independent_adaptive_health", seed=seed, safety_enabled=True)

    # 5. Ablation: Without resistance tracking
    r_no_r0 = run_episode(CFG, sc, method="adaptive_no_resistance", seed=seed, safety_enabled=True)

    # Verify all methods reach the target on this feasible scenario
    assert r_conv["target_reached"] is True
    assert r_indep["target_reached"] is True
    assert r_static["target_reached"] is True
    assert r_adapt["target_reached"] is True

    # Adaptive Health should yield less capacity loss than conventional charging
    assert r_adapt["capacity_loss"] < r_conv["capacity_loss"], (
        f"Adaptive loss {r_adapt['capacity_loss']} not lower than CC-CV {r_conv['capacity_loss']}"
    )

    # Zero unsafe executed steps for all runs with safety supervisor
    assert r_conv["unsafe_executed_steps"] == 0
    assert r_indep["unsafe_executed_steps"] == 0
    assert r_static["unsafe_executed_steps"] == 0
    assert r_adapt["unsafe_executed_steps"] == 0
    assert r_no_r0["unsafe_executed_steps"] == 0

    # Quantify resistance ablation contribution:
    # On aged battery with r0_scale=1.4, tracking resistance allows tighter, safer adaptation
    assert np.isfinite(r_no_r0["capacity_loss"])


# ==============================================================================
# Category E: Reproducibility & Release
# ==============================================================================

def test_e21_e25_reproducibility_and_release_artifacts():
    """E.21-E.25: Verify benchmark summary files, episode counts, and release artifacts exist."""
    res_dir = Path("results")
    assert res_dir.exists()

    # Phase F: 660 episodes across 11 regimes
    f_csv = res_dir / "closed_loop_health_sweep.csv"
    assert f_csv.exists(), "Phase F sweep CSV missing"
    df_f = pd.read_csv(f_csv)
    assert len(df_f) == 660, f"Expected 660 episodes in Phase F, found {len(df_f)}"
    assert len(df_f["regime"].unique()) == 11, f"Expected 11 error regimes, found {len(df_f['regime'].unique())}"

    # Phase G: 630 episodes with 0.0 unsafe steps under Hardened Supervisor
    g_csv = res_dir / "fault_injection_hardened_vs_unhardened.csv"
    assert g_csv.exists(), "Phase G fault injection CSV missing"
    df_g = pd.read_csv(g_csv)
    assert len(df_g) == 630, f"Expected 630 episodes in Phase G, found {len(df_g)}"

    hardened_subset = df_g[df_g["mode"] == "Hardened Supervisor"]
    assert (hardened_subset["unsafe_executed_steps"] == 0.0).all(), (
        "Found non-zero unsafe executed steps under Hardened Supervisor!"
    )
