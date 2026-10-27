import numpy as np, pandas as pd
from mabms.data.mode_audit import audit_file, blocks, fit_rest_cooling, load_raw


def _synth(path, tau=800.0, Ta=25.0, seed=0):
    """SYNTHETIC raw file in the NASA column layout: discharge (load cols), charge (charger col, load NaN), rest (load NaN, exponential cooling)."""
    rng = np.random.default_rng(seed); rows = []; t = 0.0; T = 25.0
    for cyc in range(3):
        for _ in range(300): t += 1; T += 0.05; rows.append((t, -1, 0, 3.7, 8.0, np.nan, T))              # discharge heats
        for _ in range(1200): t += 1; T += (Ta + 12 - T) * 0.0005; rows.append((t, 1, 0, np.nan, np.nan, 8.3, T))   # charge
        T0 = T
        for k in range(2000): t += 1; T = Ta + (T0 - Ta) * np.exp(-k / tau) + rng.normal(0, 0.02); rows.append((t, 0, 0, np.nan, np.nan, np.nan, T))
    pd.DataFrame(rows, columns=["time", "mode", "mission_type", "voltage_load", "current_load", "voltage_charger", "temperature_battery"]).to_csv(path, index=False)


def test_mode_table_blocks_and_cooling_tau(tmp_path):
    p = tmp_path / "battery00.csv"; _synth(p)
    a = audit_file(p, "battery00"); m = a["modes"].set_index("mode")
    assert m.loc[-1, "nan_voltage_charger"] == 1.0 and m.loc[1, "nan_current_load"] == 1.0 and m.loc[1, "nan_temperature_battery"] == 0.0
    b = a["blocks"].set_index("mode"); assert b.loc[0, "n_blocks"] == 3 and np.isfinite(b.loc[1, "median_dT"])
    c = a["cooling"]; assert len(c) == 3 and abs(c.tau_s.median() - 800.0) / 800.0 < 0.1
