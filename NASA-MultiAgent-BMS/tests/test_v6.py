import copy
import numpy as np
from mabms.agents.base import StateEstimate, Task
from mabms.agents.charging_agent import ChargingAgent
from mabms.agents.safety_agent import SafetyAgent
from mabms.config import load_config
from mabms.environment.battery_model import BatteryModel, ModelParams
from mabms.estimation.soc_estimator import AdaptiveR0SOCEstimator, AdaptiveSOCEstimator, OCVCurve, make_soc_estimator

cfg = load_config("configs/default.yaml")


def _est(**k):
    d = dict(soc=0.5, soh=1.0, v1=0.0, v_meas=3.8, i_meas=1.0, temp_meas=30.0, temp_pred=30.0, time_s=0.0, q_nom_ah=2.0); d.update(k)
    return StateEstimate(**d)


def test_r0_adaptive_ekf_learns_resistance_and_reduces_error():
    p = ModelParams.from_config(cfg); p.r0_ohm *= 1.4; ocv = OCVCurve(p.ocv_soc, p.ocv_v); nm = ModelParams.from_config(cfg); res = {}
    for name, cls, kw in [("cap", AdaptiveSOCEstimator, {}), ("cap+r0", AdaptiveR0SOCEstimator, {})]:
        m = BatteryModel(p, soc0=0.2, temp0_c=25, t_amb_c=25); rng = np.random.default_rng(0)
        e = cls(ocv, nm.q0_ah, nm.r0_ohm, nm.r1_ohm, nm.r1_ohm * nm.c1_f, soc0=0.2, **kw)
        for k in range(300):
            i = 0.0 if k < 5 else (2.0 if (k // 60) % 2 == 0 else 1.0)           # on/off + level changes give R0 observability
            o = m.step(i, 0.0, 10.0); e.step(i, o["voltage"] + rng.normal(0, .003), 10.0)
        res[name] = (abs(e.soc - m.soc), getattr(e, "r0_scale", 1.0))
    assert res["cap+r0"][0] < res["cap"][0] and abs(res["cap+r0"][1] - 1.4) < 0.25


def test_factory_selects_estimator():
    ocv = OCVCurve(cfg["simulator"]["ocv_soc"], cfg["simulator"]["ocv_v"]); c = copy.deepcopy(cfg); c["estimator"] = {}
    assert type(make_soc_estimator(c, ocv, 0.3)).__name__ == "SOCEstimator"
    c["estimator"] = {"adaptive_capacity": True}; assert type(make_soc_estimator(c, ocv, 0.3)).__name__ == "AdaptiveSOCEstimator"
    c["estimator"] = {"adapt_r0": True}; assert type(make_soc_estimator(c, ocv, 0.3)).__name__ == "AdaptiveR0SOCEstimator"


def test_stop_latch_prevents_chatter_near_target():
    task = Task(0.9, 1e5); seq = [0.905, 0.898, 0.905, 0.897, 0.904, 0.899]
    def run(h):
        c = copy.deepcopy(cfg); c["agents"]["charging"]["restart_hysteresis"] = h; a = ChargingAgent(c); out = []
        for s in seq: out.append(a.propose(_est(soc=s, time_s=100.0, v1=0.0), task).current_a)
        return out
    assert sum(x > 0 for x in run(0.0)) >= 2 and sum(x > 0 for x in run(0.02)) == 0         # old behaviour pulses; latch holds
    c = copy.deepcopy(cfg); c["agents"]["charging"]["restart_hysteresis"] = 0.02; a = ChargingAgent(c)
    a.propose(_est(soc=0.905), task); assert a.propose(_est(soc=0.86), task).current_a > 0       # clearly below target -> resumes


def test_r0_margin_lowers_voltage_limited_current():
    c1 = copy.deepcopy(cfg); c2 = copy.deepcopy(cfg); c2.setdefault("safety", {})["r0_margin"] = 1.6
    e = _est(soc=0.93, v_meas=4.1)
    assert SafetyAgent(c2).filter(e, 4.0, 0.0).current_a < SafetyAgent(c1).filter(e, 4.0, 0.0).current_a


def test_temp_monitor_flags_stuck_sensor_not_healthy_sensor():
    c = copy.deepcopy(cfg); c.setdefault("safety", {})["temp_monitor"] = {"enabled": True, "thr_c": 4.0, "k": 3, "gain": 0.01}
    nm = c["agents"]["nominal_model"]; dt = c["simulator"]["dt_s"]
    def run(stuck):
        sa = SafetyAgent(c); T = 30.0; flags = 0; used = []
        for k in range(150):
            reading = 30.0 if stuck else T
            est = _est(temp_meas=reading, temp_pred=reading, time_s=k * dt, soc=0.5, v_meas=3.8)
            act = sa.filter(est, 4.0, 0.0); i = act.current_a
            T += dt * (i ** 2 * (nm["r0_ohm"] * 1.4 + nm["r1_ohm"]) - nm["h_w_per_k"] * (T - 30.0) - act.cool_cmd * nm["cool_max_w"]) / nm["c_th_j_per_k"]
        return sa.n_monitor_flags, T
    f_ok, _ = run(False); f_stuck, T_true = run(True)
    assert f_ok == 0 and f_stuck > 0 and T_true > 34.0
