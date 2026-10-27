"""Per-mode audit of the RAW files (before loader.dropna): which columns are populated in discharge (-1), rest (0) and charge (1) blocks,
and what the temperature does there. Also fits Newtonian cooling to rest blocks: T(t) = Ta + (T0-Ta) exp(-t/tau).
tau = C_th / (h + cooling) is identifiable from I=0 data alone, unlike C_th and h separately from heating segments."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

WANT = ["time", "mode", "mission_type", "voltage_load", "current_load", "voltage_charger", "temperature_battery", "temperature_mosfet", "temperature_resistor"]


def load_raw(path: str | Path) -> pd.DataFrame:
    hdr = pd.read_csv(path, nrows=0).columns
    d = pd.read_csv(path, usecols=[c for c in WANT if c in hdr], low_memory=False)
    return d.apply(pd.to_numeric, errors="coerce").sort_values("time").reset_index(drop=True)


def blocks(d: pd.DataFrame, max_gap_s: float = 60.0) -> pd.DataFrame:
    t, m = d["time"].to_numpy(float), d["mode"].to_numpy(float)
    new = np.r_[True, (m[1:] != m[:-1]) | (np.diff(t) > max_gap_s)]
    bid = np.cumsum(new) - 1
    g = d.assign(block=bid).groupby("block")
    out = g.agg(mode=("mode", "first"), t0=("time", "first"), t1=("time", "last"), n=("time", "size"),
                T0=("temperature_battery", "first"), T1=("temperature_battery", "last"), Tmax=("temperature_battery", "max")).reset_index()
    out["duration_s"] = out.t1 - out.t0
    return out


def mode_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in d.groupby("mode"):
        r = {"mode": m, "rows": len(g), "share": len(g) / len(d)}
        for c in [c for c in WANT if c not in ("time", "mode") and c in d.columns]:
            r[f"nan_{c}"] = float(g[c].isna().mean())
        rows.append(r)
    return pd.DataFrame(rows)


def _cool(t, Ta, tau, T0):
    return Ta + (T0 - Ta) * np.exp(-t / tau)


def fit_rest_cooling(d: pd.DataFrame, min_block_s: float = 600.0, min_dT: float = 3.0, max_gap_s: float = 60.0) -> pd.DataFrame:
    b = blocks(d, max_gap_s); rows = []
    for _, r in b[(b["mode"] == 0) & (b.duration_s >= min_block_s)].iterrows():
        g = d[(d.time >= r.t0) & (d.time <= r.t1)].dropna(subset=["temperature_battery"])
        if len(g) < 50: continue
        t = g.time.to_numpy(float) - g.time.iloc[0]; T = g.temperature_battery.to_numpy(float); T0 = T[0]
        if abs(T0 - T[-1]) < min_dT: continue
        sel = np.linspace(0, len(t) - 1, min(len(t), 500)).astype(int)
        try:
            (Ta, tau), _ = curve_fit(lambda x, Ta, tau: _cool(x, Ta, tau, T0), t[sel], T[sel], p0=[T[-1], 800.0], bounds=([-10, 20.0], [80, 5e4]))
        except Exception:
            continue
        rmse = float(np.sqrt(np.mean((_cool(t, Ta, tau, T0) - T) ** 2)))
        rows.append({"t0": r.t0, "duration_s": r.duration_s, "T0": T0, "T_end": T[-1], "Ta_fit": Ta, "tau_s": tau, "rmse_c": rmse})
    return pd.DataFrame(rows)


def audit_file(path: str | Path, pack_id: str) -> Dict[str, Any]:
    d = load_raw(path); b = blocks(d)
    mt = mode_table(d).assign(pack_id=pack_id)
    bs = b.groupby("mode").agg(n_blocks=("block", "size"), median_dur_s=("duration_s", "median"), p10_dur_s=("duration_s", lambda x: x.quantile(.1)),
                               p90_dur_s=("duration_s", lambda x: x.quantile(.9)), median_dT=("T1", lambda x: 0.0)).reset_index().assign(pack_id=pack_id)
    ch = b[b["mode"] == 1]; bs.loc[bs["mode"] == 1, "median_dT"] = float((ch.T1 - ch.T0).median()) if len(ch) else np.nan
    cool = fit_rest_cooling(d).assign(pack_id=pack_id)
    return {"modes": mt, "blocks": bs, "cooling": cool}


# ---------------------------------------------------------------- v7: charge-protocol and end-of-data checks
def rest_tau_ta(d: pd.DataFrame) -> Tuple[float, float]:
    """Per-pack median (tau, ambient) from rest-block cooling fits."""
    c = fit_rest_cooling(d)
    return (float(c.tau_s.median()), float(c.Ta_fit.median())) if len(c) else (np.nan, np.nan)


def charge_block_stats(d: pd.DataFrame, tau: float, ta: float, min_dur_s: float = 1500.0, window_s: float = 1500.0) -> pd.DataFrame:
    """For each charge block: heating amplitude A (= P/h) from  T(t) = Ta + (T0-Ta) e^{-t/tau} + A (1 - e^{-t/tau})  (least squares over the first
    window_s seconds), observed temperature change, and the plateau fraction of voltage_charger (time within 0.03 V of the block maximum)."""
    b = blocks(d); rows = []
    for _, r in b[(b["mode"] == 1) & (b.duration_s >= min_dur_s)].iterrows():
        full = d[(d.time >= r.t0) & (d.time <= r.t1)]
        g = full[full.time <= r.t0 + window_s].dropna(subset=["temperature_battery"])
        if len(g) < 100: continue
        t = g.time.to_numpy(float) - g.time.iloc[0]; T = g.temperature_battery.to_numpy(float); T0 = T[0]
        base = 1.0 - np.exp(-t / tau); excess = T - (ta + (T0 - ta) * np.exp(-t / tau))
        A = float((excess * base).sum() / max((base * base).sum(), 1e-9))
        vc = full.voltage_charger.to_numpy(float)
        rows.append({"t0": r.t0, "duration_s": r.duration_s, "T0": T0, "dT_obs": float(T[-1] - T0), "A_heating_c": A,
                     "vc_max": float(np.nanmax(vc)), "plateau_frac": float(np.mean(vc >= np.nanmax(vc) - 0.03))})
    return pd.DataFrame(rows)


def end_of_data(d: pd.DataFrame, k: int = 6, baseline_n: int = 30) -> Dict[str, Any]:
    """Last k discharge blocks vs the median of the previous baseline_n: does the file end in an abrupt collapse (failure) or just stop?"""
    b = blocks(d); dis = b[b["mode"] == -1].reset_index(drop=True)
    if len(dis) < k + 5: return {}
    last, prev = dis.tail(k), dis.iloc[-(k + baseline_n):-k]
    def stats(blk):
        g = d[(d.time >= blk.t0) & (d.time <= blk.t1)]
        return float(g.voltage_load.iloc[:3].max()), float(g.voltage_load.min()), float(g.current_load.mean())
    base_dur = float(prev.duration_s.median()); rows = []
    for _, r in last.iterrows():
        v0, vmin, im = stats(r); rows.append({"duration_s": float(r.duration_s), "rel_dur": float(r.duration_s) / base_dur, "v_first": v0, "v_min": vmin, "i_mean": im})
    lc = b[b["mode"] == 1]
    return {"last_discharges": pd.DataFrame(rows), "baseline_dur_s": base_dur, "last_data_time_s": float(d.time.iloc[-1]),
            "last_charge_rel_dur": float(lc.duration_s.iloc[-1] / lc.duration_s.iloc[-(baseline_n + 1):-1].median()) if len(lc) > baseline_n + 1 else np.nan}
