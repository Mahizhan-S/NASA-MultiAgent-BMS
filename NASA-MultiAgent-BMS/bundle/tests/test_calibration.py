import numpy as np, pandas as pd
from mabms.config import load_config
from mabms.data.segments import build_cycle_table
from mabms.environment import calibration as cal, degradation_fit as dg
from mabms.environment.battery_model import BatteryModel, ModelParams
from mabms.estimation.soc_estimator import OCVCurve

cfg = load_config("configs/default.yaml"); cfg["data"]["reference"] = {"use_mission_type": False}


def synth_df(n_cycles=3, seed=0, loss_per_cycle=0.01, dt=5.0):
    """SYNTHETIC pack from the surrogate with known params: ref discharge (-2 A), rest, charge (+1.5 A), rest."""
    rng = np.random.default_rng(seed); p = ModelParams.from_config(cfg); rows = []; t = 0.0
    m = BatteryModel(p, soc0=1.0, loss0=0.0, temp0_c=25, t_amb_c=25)
    for c in range(n_cycles):
        m.loss = loss_per_cycle * c; m.soc = 1.0
        while m.soc > 0.02: o = m.step(-2.0, 0.0, dt); t += dt; rows.append((t, o["voltage"] + rng.normal(0, .002), -2.0, m.temp_c))
        for _ in range(4): o = m.step(0.0, 0.0, dt); t += dt; rows.append((t, o["voltage"], 0.0, m.temp_c))
        for _ in range(240): o = m.step(1.5, 0.0, dt); t += dt; rows.append((t, o["voltage"] + rng.normal(0, .002), 1.5, m.temp_c))
        for _ in range(4): o = m.step(0.0, 0.0, dt); t += dt; rows.append((t, o["voltage"], 0.0, m.temp_c))
    d = pd.DataFrame(rows, columns=["time_s", "voltage_v", "current_a", "temp_c"]); d["pack_id"] = "syn"; d["source_file"] = "syn.csv"
    return d


def _setup():
    d = synth_df(); ct = build_cycle_table(d, cfg); assert len(ct) == 3
    segs = cal.extract_segments(d, ct, cfg, n_per_type=3, stride=1)
    return d, ct, segs


def test_ocv_recovered():
    d, ct, segs = _setup(); s = cfg["simulator"]
    p = cal.VoltParams(s["r0_ohm"], s["r_ea_k"], s["r1_ohm"], s["r1_ohm"] * s["c1_f"])
    o = cal.fit_ocv_from_reference(segs, p); true = np.interp(o["soc"], s["ocv_soc"], s["ocv_v"])
    err = np.abs(np.array(o["v"]) - true)[1:-1]
    assert err.max() < 0.08 and np.all(np.diff(o["v"]) >= -1e-9)        # tolerance is loose: ~1C curve includes diffusion overpotential


def test_voltage_fit_improves_and_respects_tau_bound():
    d, ct, segs = _setup(); s = cfg["simulator"]; ocv = OCVCurve(s["ocv_soc"], s["ocv_v"])
    truth = cal.VoltParams(s["r0_ohm"], s["r_ea_k"], s["r1_ohm"], s["r1_ohm"] * s["c1_f"])
    start = cal.VoltParams(0.05, 500.0, 0.02, 40.0)
    fit = cal.fit_voltage_params(segs, ocv, start)
    r0 = cal.validate_voltage(segs, start, ocv).rmse_v.mean(); r1 = cal.validate_voltage(segs, fit, ocv).rmse_v.mean()
    assert r1 < r0 and r1 < 0.03 and fit.tau_s >= cal.TAU_MIN_S - 1e-6
    assert abs((fit.r0 + fit.r1) - (truth.r0 + truth.r1)) / (truth.r0 + truth.r1) < 0.2
    assert set(cal.validate_voltage(segs, fit, ocv).kind) >= {"ref_discharge", "charge"}


def test_thermal_fit_runs_and_not_worse():
    d, ct, segs = _setup(); s = cfg["simulator"]; ocv = OCVCurve(s["ocv_soc"], s["ocv_v"])
    p = cal.VoltParams(s["r0_ohm"], s["r_ea_k"], s["r1_ohm"], s["r1_ohm"] * s["c1_f"]); amb = {"syn": 25.0}
    c, h = cal.fit_thermal(segs, p, amb, x0=(200.0, 0.3)); assert 10 <= c <= 500 and 1e-3 <= h <= 0.5
    assert cal.validate_temperature(segs, p, c, h, amb).rmse_c.mean() <= cal.validate_temperature(segs, p, 200.0, 0.3, amb).rmse_c.mean() + 1e-9


def test_degradation_law_recovery_from_known_traces():
    rng = np.random.default_rng(0); packs = []
    k, ea, a, b = 3e-4, 4000.0, 0.8, 2.0
    for j in range(6):
        n_int = 8; n = 4000
        pk = {"ah_norm": np.full(n, 0.002) * rng.uniform(0.5, 1.5), "T_c": rng.uniform(20, 55, n), "crate": rng.uniform(0.2, 3.0, n),
              "s": rng.uniform(0, 1, n), "idx": np.sort(rng.integers(0, n_int, n)), "n_int": n_int, "soh": None}
        cum = dg.predict_cum_loss(pk, k, ea, a, b); pk["soh"] = np.r_[1.0, 1.0 - cum]; packs.append(pk)
    fit = dg.fit_variant(packs, "k_T_C_S")
    assert abs(fit["k"] - k) / k < 0.15 and abs(fit["ea"] - ea) / ea < 0.15
    base = dg.fit_variant(packs, "k_only"); e_full = dg.evaluate(packs, list("abcdef"), fit).mae_soh.mean(); e_base = dg.evaluate(packs, list("abcdef"), base).mae_soh.mean()
    assert e_full < e_base


def test_prepare_pack_shapes():
    d, ct, _ = _setup(); s = cfg["simulator"]; ocv = OCVCurve(s["ocv_soc"], s["ocv_v"])
    vp = cal.VoltParams(s["r0_ohm"], s["r_ea_k"], s["r1_ohm"], 80.0)
    pk = dg.prepare_pack(d, ct, cfg, ocv, vp, s["q0_ah"], stride=2)
    assert pk["n_int"] == len(ct) - 1 and len(pk["idx"]) == len(pk["ah_norm"]) > 0 and pk["idx"].max() < pk["n_int"]
    assert np.all(np.diff(dg.predict_cum_loss(pk, 1e-3, 0, 0, 0)) >= 0)
