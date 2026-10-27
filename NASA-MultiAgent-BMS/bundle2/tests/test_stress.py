import numpy as np
from mabms.config import load_config
from mabms.evaluation.stress import FAULTS, apply_fault, make_stress_scenarios, run_stress_episode

cfg = load_config("configs/default.yaml")


def test_aggressive_proposer_is_unsafe_and_safety_reduces_executed_violations():
    sc = make_stress_scenarios(1, 0)[0]
    on = run_stress_episode(cfg, sc, "aggressive_3.5C", "none", True, 1); off = run_stress_episode(cfg, sc, "aggressive_3.5C", "none", False, 1)
    assert off["unsafe_prop_true_steps"] > 0 and on["interventions"] > 0
    assert on["unsafe_executed_steps"] <= off["unsafe_executed_steps"]


def test_nan_temperature_triggers_fallback_with_safety_on():
    sc = make_stress_scenarios(1, 3)[0]; hit = 0
    for seed in range(4):
        r = run_stress_episode(cfg, sc, "independent_deadline", "t_dropout_5pct", True, seed); hit += r["fallbacks"]
    assert hit > 0


def test_faults_apply_as_declared():
    rng = np.random.default_rng(0); o = {"voltage_v": 3.9, "temp_c": 40.0}
    assert apply_fault(o, "v_bias_+0.1V", rng, 30.0)["voltage_v"] == 4.0 and apply_fault(o, "t_bias_-5C", rng, 30.0)["temp_c"] == 35.0
    assert apply_fault(o, "t_stuck", rng, 30.0)["temp_c"] == 30.0 and set(FAULTS) >= {"none", "t_stuck"}
