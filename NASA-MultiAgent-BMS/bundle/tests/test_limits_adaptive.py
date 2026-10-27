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
