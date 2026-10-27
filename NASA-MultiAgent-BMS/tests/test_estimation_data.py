import numpy as np, pandas as pd
from mabms.config import load_config
from mabms.data.loader import load_all
from mabms.data.segments import build_cycle_table
from mabms.data.splits import split_packs, grouped_folds
from mabms.estimation.soc_estimator import SOCEstimator, OCVCurve
from mabms.estimation import soh_estimator as soh, temperature_predictor as tp, rul_predictor as rul
from mabms.environment.battery_model import BatteryModel, ModelParams

cfg = load_config("configs/default.yaml")

def synth_pack(path, n_cycles=30, fade=0.012, seed=0):
    """SYNTHETIC file in the assumed raw schema (discharge current positive). For code tests only."""
    rng = np.random.default_rng(seed); p = ModelParams.from_config(cfg); rows = []; t = 0.0
    for c in range(n_cycles):
        m = BatteryModel(p, soc0=1.0, loss0=fade * c, temp0_c=25, t_amb_c=25)
        for k in range(100000):
            o = m.step(-2.0, 0.0, 5.0); t += 5.0
            rows.append((t, o["voltage"] + rng.normal(0, .002), 2.0, m.temp_c + rng.normal(0, .1)))
            if m.soc <= 0.02: break
        for _ in range(5): t += 5.0; rows.append((t, 3.5, 0.0, m.temp_c))      # rest
        m.soc = 0.0
    pd.DataFrame(rows, columns=["time", "voltage_load", "current_load", "temperature_battery"]).to_csv(path, index=False)

def test_pipeline_on_synthetic(tmp_path):
    for k in range(4): synth_pack(tmp_path / f"pack{k}.csv", fade=0.010 + 0.004 * k, seed=k)
    c = load_all(tmp_path, cfg); assert len(c) == 4
    tabs = {p: build_cycle_table(d, cfg) for p, d in c.items()}
    for p, t in tabs.items():
        assert len(t) >= 25 and abs(t["soh"].iloc[:3].median() - 1.0) < 1e-9 and t["soh"].iloc[-1] < t["soh"].iloc[0]
    tab = pd.concat(tabs.values(), ignore_index=True)
    r = soh.grouped_cv(tab, "ridge", 2); assert np.isfinite(r["overall"]["mae"])
    t = tp.grouped_cv(c, 5, 10, 5, 2); assert t["persistence"]["overall"]["rmse"] >= 0
    rt, cens = rul.build_rul_table(tabs, 0.8, 10); assert isinstance(cens, pd.DataFrame)

def test_split_is_pack_level_disjoint():
    ids = [f"p{i}" for i in range(26)]; s = split_packs(ids, .2, .2, 0)
    assert not (set(s["train"]) & set(s["test"])) and not (set(s["train"]) & set(s["val"])) and len(set().union(*s.values())) == 26
    g = np.repeat(ids, 5)
    for tr, te in grouped_folds(g, 4): assert not (set(g[tr]) & set(g[te]))

def test_ekf_converges_from_wrong_initial_soc():
    p = ModelParams.from_config(cfg); m = BatteryModel(p, soc0=0.7, temp0_c=25, t_amb_c=25); ocv = OCVCurve(p.ocv_soc, p.ocv_v)
    ekf = SOCEstimator(ocv, p.q0_ah, p.r0_ohm, p.r1_ohm, p.r1_ohm * p.c1_f, soc0=0.4); rng = np.random.default_rng(0)
    for k in range(600):
        o = m.step(-1.0 if k % 2 else 0.5, 0.0, 10.0); ekf.step(-1.0 if k % 2 else 0.5, o["voltage"] + rng.normal(0, .003), 10.0)
    assert abs(ekf.soc - m.soc) < 0.05


def test_stats_and_estimator_comparison(tmp_path):
    from mabms.evaluation.stats import holm, paired_compare
    from mabms.evaluation.compare_estimators import compare_soh, compare_temperature
    assert holm([0.01, 0.04, 0.03])[0] == 0.03 and max(holm([0.5, 0.9])) <= 1.0
    df = pd.DataFrame({"scenario_idx": list(range(10)) * 2, "method": ["a"] * 10 + ["b"] * 10, "x": list(np.arange(10)) + list(np.arange(10) + 1.0)})
    r = paired_compare(df, "x", "a", "b"); assert abs(r["mean_diff"] + 1.0) < 1e-9 and r["wilcoxon_p"] < 0.01
    for k in range(6): synth_pack(tmp_path / f"p{k}.csv", n_cycles=20, fade=0.008 + 0.003 * k, seed=k)
    c = load_all(tmp_path, cfg); tab = pd.concat([build_cycle_table(d, cfg) for d in c.values()], ignore_index=True)
    ids = sorted(c); out = compare_soh(tab, ids[:4], ids[4:], 2, ("mean", "ridge", "hgb"))
    assert len(out["cv"]) == 6 and len(out["test"]) == 6 and "ridge|throughput_only" in set(out["cv"]["config"])
    tt = compare_temperature(c, ids[:4], ids[4:], {"lags": 5, "horizon_steps": 10, "stride": 20}, 2)
    assert set(tt["test"]["model"]) == {"persistence", "ridge", "hgb"}


def _synth_mixed(path, n_checks=12, seed=0):
    """SYNTHETIC: reference (2 A, mission 0) checks interleaved with high-rate (6 A, mission 1) cycles; fade per throughput; one spike + zero-temp rows."""
    rng = np.random.default_rng(seed); p = ModelParams.from_config(cfg); rows = []; t = 0.0; loss = 0.0
    for c in range(n_checks):
        for cur, mt in [(-6.0, 1), (-2.0, 0)]:
            m = BatteryModel(p, soc0=1.0, loss0=loss, temp0_c=25, t_amb_c=25)
            while m.soc > 0.02:
                o = m.step(cur, 0.0, 5.0); t += 5.0; rows.append((t, o["voltage"], -cur, m.temp_c, mt))
            loss += 0.02
            for _ in range(3): t += 5.0; rows.append((t, 3.5, 0.0, m.temp_c, mt))
    df = pd.DataFrame(rows, columns=["time", "voltage_load", "current_load", "temperature_battery", "mission_type"])
    df.loc[50, "voltage_load"] = 4.6 - 0.0 + 0.4          # isolated spike
    df.loc[60, "temperature_battery"] = 0.0              # invalid temp
    df.to_csv(path, index=False)


def test_reference_only_throughput_and_cleaning(tmp_path):
    from mabms.data.cleaning import clean_signals
    from mabms.estimation import rul_predictor as rul
    _synth_mixed(tmp_path / "mixed0.csv"); _synth_mixed(tmp_path / "mixed1.csv", seed=1)
    c = load_all(tmp_path, cfg)
    for pid, d in c.items():
        n0 = len(d); d2 = clean_signals(d, cfg)
        assert d.attrs.get("n_dropped", 0) >= 2 or len(d2) <= n0 - 2          # spike and zero-temp removed
        t = build_cycle_table(d2, cfg)
        assert len(t) == 12 and (t["mission_type"] == 0).all()                      # only reference discharges
        assert t["throughput_ah"].is_monotonic_increasing and t["throughput_ah"].iloc[-1] > 20
        assert t["soh"].iloc[-1] < t["soh"].iloc[0]
    tabs = {p: build_cycle_table(clean_signals(d, cfg), cfg) for p, d in c.items()}
    tbl, cens = rul.build_rul_table_v2(tabs, 0.85, min_refs=6)
    assert len(tbl) > 0 and "global_lin_rul" in tbl
    o = rul.rul_study_v2(tbl, tabs, cens, 0.85, 2); assert set(o["eol_packs"]["model"]) == set(rul.MODELS_V2)


def test_env_tolerance_and_shortfall():
    from mabms.environment.battery_env import BatteryEnv, Scenario
    c2 = load_config("configs/default.yaml"); c2["simulator"]["target_tolerance"] = 0.01
    env = BatteryEnv(c2); env.reset(0, Scenario(soc0=0.78, target_soc=0.8, deadline_s=1e5))
    done = False
    while not done: _, _, te, tr, _ = env.step([1.0, 0.0]); done = te or tr
    m = env.get_metrics(); assert m["target_reached"] and m["shortfall"] <= 0.0105 and m["time_to_target_penalized_s"] == m["time_to_target_s"]
    env.reset(0, Scenario(soc0=0.2, target_soc=0.9, deadline_s=100.0)); done = False
    while not done: _, _, te, tr, _ = env.step([0.1, 0.0]); done = te or tr
    m = env.get_metrics(); assert (not m["target_reached"]) and m["time_to_target_penalized_s"] == 100.0 and m["shortfall"] > 0.5


def test_spike_filter_catches_relay_sized_spike_keeps_steps():
    from mabms.data.cleaning import isolated_spike_mask
    # per-cell units: rest 4.175 -> relay spike 4.3285 (+0.15) -> loaded 3.66 -> loaded; plus a genuine load step elsewhere
    v = np.array([4.175, 4.176, 4.175, 4.3285, 3.66, 3.65, 3.65, 3.64, 4.17, 4.17])
    m = isolated_spike_mask(v, 0.1)
    assert m[3] and m.sum() == 1                                   # only the spike; load-step edges preserved
    assert not isolated_spike_mask(v, 0.2)[3]                      # the old 0.2 default would have missed it

