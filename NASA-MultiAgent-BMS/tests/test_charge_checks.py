import numpy as np, pandas as pd
from mabms.data.mode_audit import charge_block_stats, end_of_data, load_raw, rest_tau_ta


def _synth(path, tau=800.0, Ta=25.0, A=5.0, n=40, fail=False, seed=0):
    """SYNTHETIC raw file: discharge (hot) -> rest 600+ s (exp. cooling) -> charge 2000 s with known heating amplitude A -> ..."""
    rng = np.random.default_rng(seed); rows = []; t = 0.0; T = 25.0
    for c in range(n):
        dur = 500 if not (fail and c >= n - 3) else 120                          # failure: last 3 discharges collapse
        for _ in range(dur): t += 1; T += 0.08; rows.append((t, -1, 1, 7.0 if not (fail and c >= n - 3) else 3.0, 16.0, np.nan, T))
        T0 = T
        for k in range(900): t += 1; T = Ta + (T0 - Ta) * np.exp(-k / tau) + rng.normal(0, .02); rows.append((t, 0, 1, np.nan, np.nan, np.nan, T))
        T0 = T
        for k in range(2000):
            t += 1; T = Ta + (T0 - Ta) * np.exp(-k / tau) + A * (1 - np.exp(-k / tau)) + rng.normal(0, .02)
            rows.append((t, 1, 1, np.nan, np.nan, 8.0 + 0.63 * min(k / 1500, 1), T))
    pd.DataFrame(rows, columns=["time", "mode", "mission_type", "voltage_load", "current_load", "voltage_charger", "temperature_battery"]).to_csv(path, index=False)


def test_charge_heating_amplitude_recovered(tmp_path):
    p = tmp_path / "b.csv"; _synth(p, A=5.0); d = load_raw(p); tau, ta = rest_tau_ta(d)
    assert abs(tau - 800) / 800 < 0.15 and abs(ta - 25) < 1.0
    cb = charge_block_stats(d, tau, ta); assert len(cb) > 20 and abs(cb.A_heating_c.median() - 5.0) < 1.0
    cb0 = charge_block_stats(*(lambda dd: (dd, *rest_tau_ta(dd)))(_load0(tmp_path))); assert abs(cb0.A_heating_c.median()) < 1.0
    assert cb.plateau_frac.median() > 0.1


def _load0(tmp_path):
    p = tmp_path / "b0.csv"; _synth(p, A=0.0); return load_raw(p)


def test_end_of_data_flags_collapse(tmp_path):
    p1, p2 = tmp_path / "ok.csv", tmp_path / "fail.csv"; _synth(p1, n=45); _synth(p2, n=45, fail=True)
    ok, bad = end_of_data(load_raw(p1)), end_of_data(load_raw(p2))
    assert ok["last_discharges"].rel_dur.iloc[-1] > 0.9 and bad["last_discharges"].rel_dur.iloc[-1] < 0.5 and bad["last_discharges"].v_min.min() < 4.0
