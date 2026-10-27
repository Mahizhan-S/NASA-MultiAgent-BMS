import numpy as np, pytest
from mabms.config import load_config
from mabms.environment.battery_env import BatteryEnv, Scenario, make_scenarios
from mabms.agents.base import StateEstimate
from mabms.agents.safety_agent import SafetyAgent
from mabms.evaluation.run_independent import run_episode

cfg = load_config("configs/default.yaml")

def rollout(i, cool, n=60, sc=None):
    env = BatteryEnv(cfg); env.reset(seed=0, scenario=sc or Scenario(soc0=0.3, deadline_s=1e6, target_soc=1.0))
    for _ in range(n): env.step([i, cool])
    return env

def test_more_current_more_heat_and_soc():
    a, b = rollout(1.0, 0.0), rollout(3.0, 0.0)
    assert b.model.temp_c > a.model.temp_c and b.model.soc > a.model.soc and b.model.loss > a.model.loss

def test_cooling_lowers_temp():
    assert rollout(3.0, 1.0).model.temp_c < rollout(3.0, 0.0).model.temp_c

def test_zero_current_keeps_soc():
    assert abs(rollout(0.0, 0.0).model.soc - 0.3) < 1e-9

def test_invalid_action_is_noop():
    env = BatteryEnv(cfg); env.reset(seed=0, scenario=Scenario(soc0=0.3)); env.step([np.nan, 0.0])
    assert env.model.soc == pytest.approx(0.3)

def test_negative_current_clipped():
    env = BatteryEnv(cfg); env.reset(seed=0, scenario=Scenario(soc0=0.3)); env.step([-5.0, 0.0])
    assert env.model.soc >= 0.3

def _est(**k):
    d = dict(soc=0.5, soh=1.0, v1=0.0, v_meas=3.8, i_meas=1.0, temp_meas=30.0, temp_pred=30.0, time_s=0.0, q_nom_ah=2.0); d.update(k)
    return StateEstimate(**d)

def test_safety_invalid_state_fallback():
    s = SafetyAgent(cfg); a = s.filter(_est(soc=float("nan")), 1.0, 0.2)
    assert a.fallback and a.current_a == 0.0 and a.cool_cmd == 1.0 and len(s.log) == 1

def test_safety_limits_current_and_voltage_and_temp():
    s = SafetyAgent(cfg)
    from mabms.limits import current_limit_a
    assert s.filter(_est(), 50.0, 0.0).current_a <= current_limit_a(cfg) + 1e-9
    assert s.filter(_est(soc=0.99, v_meas=4.19), 3.0, 0.0).current_a < 3.0
    a = s.filter(_est(temp_meas=44.9, temp_pred=45.5), 2.0, 0.0)
    assert a.current_a < 0.2 and a.cool_cmd == 1.0
    assert s.filter(_est(soc=1.0), 1.0, 0.0).current_a == 0.0

def test_safety_disabled_passes_through():
    s = SafetyAgent(cfg, enabled=False); assert s.filter(_est(), 50.0, 0.0).current_a == 50.0

def test_episodes_run_and_safety_reduces_violations():
    sc = make_scenarios(1, 0, "test_unseen")[0]
    for m in ["conventional", "independent", "independent_health_cap"]:
        r = run_episode(cfg, sc, m, seed=1); assert r["steps"] > 0 and r["soc_est_mae"] < 0.2
    # ablation: with safety disabled an aggressive charger should not do better on violations
    on = run_episode(cfg, sc, "independent", 1, True); off = run_episode(cfg, sc, "independent", 1, False)
    assert off["unsafe_executed_steps"] >= on["unsafe_executed_steps"]
