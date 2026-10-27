"""Segmentation into charge/discharge/rest segments and derived (NOT measured) labels.
Convention: current_a > 0 is charge. Capacity/SOH are DERIVED labels (coulomb counting).
v2: SOH labels use REFERENCE discharges only (mission_type == reference value) to remove load-dependent (rate-capacity) bias;
    ageing axis = cumulative |Ah| throughput and operating time, not the index of reference cycles."""
from __future__ import annotations
import warnings
from typing import Any, Dict
import numpy as np
import pandas as pd


def _trapz(y: np.ndarray, t: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))


def label_segments(df: pd.DataFrame, i_thr: float, max_gap_s: float) -> pd.DataFrame:
    d = df.sort_values("time_s").reset_index(drop=True).copy()
    t, i = d["time_s"].to_numpy(), d["current_a"].to_numpy()
    code = np.where(i > i_thr, 1, np.where(i < -i_thr, -1, 0))  # 1 charge, -1 discharge, 0 rest
    dt = np.r_[0.0, np.diff(t)]
    gap = dt > max_gap_s
    change = np.r_[False, code[1:] != code[:-1]] | gap
    d["seg_code"] = code
    d["seg_id"] = np.cumsum(change)
    dt_ok = np.where(gap, 0.0, dt)                         # never integrate across data gaps
    d["cum_throughput_ah"] = np.cumsum(np.abs(i) * dt_ok) / 3600.0
    d["cum_op_time_s"] = np.cumsum(dt_ok)
    return d


def segment_table(d: pd.DataFrame, first_window_s: float = 60.0) -> pd.DataFrame:
    rows = []
    has_m = "mission_type" in d.columns
    for sid, g in d.groupby("seg_id"):
        t = g["time_s"].to_numpy(); i = g["current_a"].to_numpy()
        v = g["voltage_v"].to_numpy(); T = g["temp_c"].to_numpy()
        w = g[g["time_s"] <= t[0] + first_window_s]
        r = {
            "pack_id": g["pack_id"].iloc[0], "seg_id": int(sid), "seg_code": int(g["seg_code"].iloc[0]),
            "t_start": t[0], "duration_s": t[-1] - t[0], "n": len(g),
            "ah": abs(_trapz(i, t)) / 3600.0, "v_start": v[0], "v_end": v[-1], "v_mean": v.mean(),
            "i_mean": i.mean(), "temp_mean": T.mean(), "temp_max": T.max(), "temp_rise": T[-1] - T[0],
            "throughput_ah": float(g["cum_throughput_ah"].iloc[0]), "op_time_s": float(g["cum_op_time_s"].iloc[0]),
            "w_v0": w["voltage_v"].iloc[0], "w_v_end": w["voltage_v"].iloc[-1],
            "w_dv": w["voltage_v"].iloc[-1] - w["voltage_v"].iloc[0],
            "w_i_mean": w["current_a"].mean(), "w_temp_mean": w["temp_c"].mean(),
            "w_v_drop_step": float(abs(v[min(1, len(v) - 1)] - v[0])),
        }
        if has_m:
            r["mission_type"] = float(pd.to_numeric(g["mission_type"], errors="coerce").mode().iloc[0]) if g["mission_type"].notna().any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def build_cycle_table(df_pack: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Reference full-discharge capacity per check + SOH relative to the pack's own early reference capacity."""
    sc, fc = cfg["data"]["segmentation"], cfg["data"]["full_discharge"]
    ref = cfg["data"].get("reference", {"use_mission_type": True, "mission_type_value": 0})
    d = label_segments(df_pack, sc["i_threshold_a"], sc["max_gap_s"])
    st = segment_table(d)
    full = st[(st.seg_code == -1) & (st.duration_s >= fc["min_duration_s"])
              & (st.v_start >= fc["v_start_min"]) & (st.v_end <= fc["v_end_max"])].copy()
    if ref.get("use_mission_type", True):
        if "mission_type" in full.columns:
            full = full[full["mission_type"] == ref.get("mission_type_value", 0)]
        else:
            warnings.warn("reference filter requested but no mission_type column: SOH labels are load-confounded")
    full = full.sort_values("t_start").reset_index(drop=True)
    if full.empty:
        return full
    full["ref_idx"] = np.arange(len(full)); full["cycle_idx"] = full["ref_idx"]     # index among REFERENCE checks only
    full["op_time_h"] = full["op_time_s"] / 3600.0
    full["capacity_ah"] = full["ah"]
    refcap = full["capacity_ah"].iloc[: fc["n_reference_cycles"]].median()
    full["cap_ref_ah"] = refcap
    full["soh"] = full["capacity_ah"] / refcap     # DERIVED label
    return full
