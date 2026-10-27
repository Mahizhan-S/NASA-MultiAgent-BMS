import copy
import numpy as np
from mabms.config import load_config
from mabms.environment.battery_env import BatteryEnv, Scenario
from mabms.environment.battery_model import BatteryModel, ModelParams
from mabms.estimation.soc_estimator import AdaptiveSOCEstimator, OCVCurve, SOCEstimator
from mabms.evaluation.objective import ensure_scales
from mabms.evaluation.run_independent import run_episode
from mabms.limits import current_limit_a

cfg = load_config("configs/default.yaml")


def test_limit_is_absolute_and_independent_of_cell_capacity():
    assert abs(current_limit_a(cfg) - cfg["limits"]["i_max_c"] * cfg["simulator"]["q0_ah"]) < 1e-9
    c2 = copy.deepcopy(cfg); c2["limits"]["i_max_a"] = 3.0; assert current_limit_a(c2) == 3.0


def test_low_capacity_cell_no_current_violation_with_safety_on():
    sc = Scenario(soc0=0.2, target_soc=0.9, deadline_s=2500, q_scale=0.85, loss0=0.0)       # tight deadline -> aggressive current
    bp = {"c_rate": 2.0, "v_cv_offset": 0.0, "kp": 0.3, "setpoint_c": 38.0, "deadline_gain": 1.5}
    r = run_episode(cfg, sc, "single_agent", seed=1, policy_params=bp)
    assert r["sim_violations_i"] == 0


def test_env_counts_violation_when_safety_off_and_limit_exceeded():
    env = BatteryEnv(cfg); env.reset(0, Scenario(soc0=0.3, target_soc=1.0, deadline_s=1e5)); lim = current_limit_a(cfg)
    env.step([lim * 1.2, 0.0]); assert env.get_metrics()["sim_violations_i"] == 1


def test_adaptive_ekf_removes_capacity_bias():
    p = ModelParams.from_config(cfg); p.q0_ah *= 0.85                               # real cell has 85% of the nominal capacity
    ocv = OCVCurve(p.ocv_soc, p.ocv_v); rng = np.random.default_rng(0); out = {}
    for name, cls, kw in [("fixed", SOCEstimator, {}), ("adaptive", AdaptiveSOCEstimator, {"q_b": 1e-7})]:
        m = BatteryModel(p, soc0=0.2, temp0_c=25, t_amb_c=25); rng = np.random.default_rng(0)
        e = cls(ocv, 2.0, p.r0_ohm, p.r1_ohm, p.r1_ohm * p.c1_f, soc0=0.2, **kw)           # estimator believes nominal 2.0 Ah
        for k in range(300):
            o = m.step(2.0, 0.0, 10.0); e.step(2.0, o["voltage"] + rng.normal(0, .003), 10.0)
        out[name] = abs(e.soc - m.soc)
    assert out["adaptive"] < 0.5 * out["fixed"] and out["adaptive"] < 0.03


def test_ensure_scales_sets_positive_numbers():
    c2 = copy.deepcopy(cfg); c2["objective"]["d_scale"] = "auto"; c2["objective"]["e_scale"] = "auto"
    s = ensure_scales(c2, n=3); assert s["d_scale"] > 0 and s["e_scale"] > 0 and c2["objective"]["d_scale"] == s["d_scale"]


def test_adaptive_health_agent_derates_and_preserves_deadline():
    from mabms.agents.health_agent import HealthAgent
    from mabms.agents.base import StateEstimate, Task

    ha = HealthAgent(cfg)
    # Case A: Ample time (relaxed deadline)
    # Fresh cell:
    est_fresh = StateEstimate(soc=0.3, soh=1.0, v1=0.0, v_meas=3.7, i_meas=1.0,
                              temp_meas=25.0, temp_pred=25.0, time_s=0.0, q_nom_ah=2.0, r0_scale=1.0)
    task_relaxed = Task(target_soc=0.8, deadline_s=7200.0) # 2 hours
    hp_fresh = ha.propose(est_fresh, task_relaxed)

    # Aged, hot, high resistance cell:
    est_aged = StateEstimate(soc=0.3, soh=0.75, v1=0.0, v_meas=3.7, i_meas=1.0,
                             temp_meas=40.0, temp_pred=40.5, time_s=0.0, q_nom_ah=2.0, r0_scale=1.8)
    hp_aged = ha.propose(est_aged, task_relaxed)

    # Health agent must derate or stay gentle for the aged/hot cell
    assert hp_aged.max_c_rate <= hp_fresh.max_c_rate
    # Both must respect min_c
    assert hp_aged.max_c_rate >= cfg["agents"]["health"]["min_c"]

    # Case B: Urgent deadline on aged cell (avoid starvation)
    task_urgent = Task(target_soc=0.8, deadline_s=1200.0) # 20 mins for 50% SOC -> ~1.5 C required
    hp_urgent = ha.propose(est_aged, task_urgent)
    # Must expand to meet deadline rather than starving the cell
    assert hp_urgent.max_c_rate > hp_aged.max_c_rate
    assert hp_urgent.max_c_rate <= cfg["limits"]["i_max_c"]


def test_adaptive_health_agent_phase_b_comprehensive_acceptance():
    from mabms.agents.health_agent import HealthAgent
    from mabms.agents.base import StateEstimate, Task

    ha = HealthAgent(cfg)
    ha.reset()

    # 1. Fresh vs Aged cell
    e_fresh = StateEstimate(soc=0.4, soh=1.0, v1=0.0, v_meas=3.8, i_meas=1.0,
                            temp_meas=25.0, temp_pred=25.0, time_s=0.0, q_nom_ah=2.0, r0_scale=1.0, soh_sigma=0.01)
    e_aged = StateEstimate(soc=0.4, soh=0.72, v1=0.0, v_meas=3.8, i_meas=1.0,
                           temp_meas=25.0, temp_pred=25.0, time_s=0.0, q_nom_ah=2.0, r0_scale=1.0, soh_sigma=0.01)
    t_relaxed = Task(target_soc=0.8, deadline_s=5000.0)
    hp_fresh = ha.propose(e_fresh, t_relaxed)
    ha.reset()
    hp_aged = ha.propose(e_aged, t_relaxed)
    assert hp_aged.max_c_rate <= hp_fresh.max_c_rate

    # 2. Elevated resistance derating
    ha.reset()
    e_r0_high = StateEstimate(soc=0.4, soh=1.0, v1=0.0, v_meas=3.8, i_meas=1.0,
                              temp_meas=25.0, temp_pred=25.0, time_s=0.0, q_nom_ah=2.0, r0_scale=2.2, soh_sigma=0.01)
    hp_r0_high = ha.propose(e_r0_high, t_relaxed)
    ha.reset()
    hp_r0_norm = ha.propose(e_fresh, t_relaxed)
    assert hp_r0_high.max_c_rate < hp_r0_norm.max_c_rate

    # 3. High temperature derating
    ha.reset()
    e_hot = StateEstimate(soc=0.4, soh=1.0, v1=0.0, v_meas=3.8, i_meas=1.0,
                          temp_meas=42.0, temp_pred=42.5, time_s=0.0, q_nom_ah=2.0, r0_scale=1.0, soh_sigma=0.01)
    hp_hot = ha.propose(e_hot, t_relaxed)
    ha.reset()
    hp_cool = ha.propose(e_fresh, t_relaxed)
    assert hp_hot.max_c_rate < hp_cool.max_c_rate

    # 4. Feasibility Classification: FEASIBLE, AT_RISK, INFEASIBLE
    ha.reset()
    # (a) FEASIBLE: relaxed deadline
    assert hp_fresh.feasibility_status == "FEASIBLE"

    # (b) AT_RISK: tight deadline that requires current above health ceiling but within hardware limit
    t_at_risk = Task(target_soc=0.9, deadline_s=1500.0)
    ha.reset()
    hp_risk = ha.propose(e_aged, t_at_risk)
    assert hp_risk.feasibility_status in ("AT_RISK", "FEASIBLE")

    # (c) INFEASIBLE: physically impossible deadline (e.g. 50% charge in 10 seconds -> ~180 C)
    t_impossible = Task(target_soc=0.9, deadline_s=10.0)
    ha.reset()
    hp_infeasible = ha.propose(e_fresh, t_impossible)
    assert hp_infeasible.feasibility_status == "INFEASIBLE"
    assert hp_infeasible.max_c_rate <= cfg["limits"]["i_max_c"]

    # 5. Slew-rate limiting across rapid successive time steps
    ha.reset()
    hp_step1 = ha.propose(e_fresh, t_relaxed) # relaxed -> low C
    e_fresh_step2 = copy.deepcopy(e_fresh)
    e_fresh_step2.time_s = 1.0 # 1 second later
    # Sudden urgent request
    hp_step2 = ha.propose(e_fresh_step2, t_impossible)
    # The jump between step 1 and step 2 must be bounded by slew rate limit * dt
    slew_limit = ha.slew_rate_c_per_s * 1.0
    assert (hp_step2.max_c_rate - hp_step1.max_c_rate) <= slew_limit + 1e-4

    # 6. SOH estimation uncertainty handling
    ha.reset()
    e_soh_certain = StateEstimate(soc=0.4, soh=0.85, v1=0.0, v_meas=3.8, i_meas=1.0,
                                  temp_meas=25.0, temp_pred=25.0, time_s=0.0, q_nom_ah=2.0, r0_scale=1.0, soh_sigma=0.005)
    e_soh_uncertain = StateEstimate(soc=0.4, soh=0.85, v1=0.0, v_meas=3.8, i_meas=1.0,
                                    temp_meas=25.0, temp_pred=25.0, time_s=0.0, q_nom_ah=2.0, r0_scale=1.0, soh_sigma=0.08)
    hp_certain = ha.propose(e_soh_certain, t_relaxed)
    ha.reset()
    hp_uncertain = ha.propose(e_soh_uncertain, t_relaxed)
    assert hp_uncertain.max_c_rate <= hp_certain.max_c_rate

    # 7. Edge Cases & Deterministic Fallbacks
    ha.reset()
    # (a) Zero remaining time (expired deadline)
    t_zero_time = Task(target_soc=0.8, deadline_s=0.0)
    hp_zero_time = ha.propose(e_fresh, t_zero_time)
    assert hp_zero_time.feasibility_status == "INFEASIBLE"
    assert np.isfinite(hp_zero_time.max_c_rate)

    # (b) Already reached target SOC (Delta SOC <= 0)
    t_done = Task(target_soc=0.3, deadline_s=3600.0) # current soc is 0.4
    hp_done = ha.propose(e_fresh, t_done)
    assert hp_done.feasibility_status == "FEASIBLE"
    assert hp_done.max_c_rate >= cfg["agents"]["health"]["min_c"]

    # (c) Invalid / NaN state estimate -> must trigger fallback cleanly without crash
    e_nan = StateEstimate(soc=float("nan"), soh=1.0, v1=0.0, v_meas=3.8, i_meas=1.0,
                          temp_meas=25.0, temp_pred=25.0, time_s=0.0, q_nom_ah=2.0)
    hp_fallback = ha.propose(e_nan, t_relaxed)
    assert hp_fallback.is_fallback is True
    assert hp_fallback.feasibility_status == "INFEASIBLE"
    assert np.isfinite(hp_fallback.max_c_rate)
    assert np.isfinite(hp_fallback.v_cv)

    # 8. Physical validity check
    assert hp_fresh.predicted_dloss >= 0.0
    assert not np.isnan(hp_fresh.max_c_rate)
    assert not np.isinf(hp_fresh.max_c_rate)
    assert hp_fresh.max_c_rate >= 0.0
