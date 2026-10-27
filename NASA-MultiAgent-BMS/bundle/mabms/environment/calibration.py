"""Simulator calibration on REAL traces (per-cell-equivalent). Everything is fitted on TRAIN packs and reported on VAL (test only on demand).
Stage 0: OCV curve from reference discharges (IR-corrected).  Stage 1: voltage params (R0(T), R1, tau>=tau_min, cable R for charger voltage).
Stage 2: thermal params (C_th, h) with measured T used for R0(T).  Limitations (document them): ~1C reference discharge still contains
diffusion overpotential (OCV biased low at low SOC); single RC branch; charger-voltage validation includes cable drop; ambient proxy."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter
from sklearn.isotonic import IsotonicRegression
from ..data.segments import label_segments, segment_table
from ..estimation.soc_estimator import OCVCurve

K0 = 273.15
TAU_MIN_S = 30.0


@dataclass
class VoltParams:
    r0: float; r_ea_k: float; r1: float; tau_s: float; r_cable: float = 0.0

    def vec(self) -> np.ndarray:
        return np.array([self.r0, self.r_ea_k, self.r1, self.tau_s, self.r_cable], float)

    @property
    def c1(self) -> float:
        return self.tau_s / self.r1


def r0_of_T(r0, r_ea_k, T_c, t_ref_c: float = 25.0):
    return r0 * np.exp(r_ea_k * (1.0 / (np.asarray(T_c, float) + K0) - 1.0 / (t_ref_c + K0)))


# ---------------------------------------------------------------- segments
def extract_segments(df_pack: pd.DataFrame, ct: pd.DataFrame, cfg: Dict[str, Any], n_per_type: int = 4,
                     min_duration_s: float = 300.0, stride: int = 5) -> List[Dict[str, Any]]:
    """Reference discharges (matched to the cycle table), other discharges, and charges, evenly spread over the pack's life.
    If column v_charger_cell exists (mostly non-NaN) it is used as the voltage of charge segments (flag v_is_charger)."""
    sc = cfg["data"]["segmentation"]
    d = label_segments(df_pack, sc["i_threshold_a"], sc["max_gap_s"]); st = segment_table(d)
    ref_t = set(np.round(ct["t_start"].to_numpy(), 3))
    st["is_ref"] = (st.seg_code == -1) & np.round(st.t_start, 3).isin(ref_t)
    kinds = {"ref_discharge": st[st.is_ref],
             "alt_discharge": st[(st.seg_code == -1) & (~st.is_ref) & (st.duration_s >= min_duration_s)],
             "charge": st[(st.seg_code == 1) & (st.duration_s >= min_duration_s)]}
    has_chg = "v_charger_cell" in d.columns and d["v_charger_cell"].notna().mean() > 0.5
    ct_tp, ct_soh = ct["throughput_ah"].to_numpy(), ct["soh"].to_numpy(); cap_ref = float(ct["cap_ref_ah"].iloc[0])
    groups = dict(tuple(d.groupby("seg_id")))
    out = []
    for kind, tab in kinds.items():
        if tab.empty: continue
        idx = np.unique(np.linspace(0, len(tab) - 1, min(n_per_type, len(tab))).round().astype(int))
        for _, r in tab.iloc[idx].iterrows():
            g = groups[int(r.seg_id)].iloc[::stride]
            if len(g) < 20: continue
            use_chg = kind == "charge" and has_chg and g["v_charger_cell"].notna().mean() > 0.9
            out.append({"kind": kind, "pack_id": r.pack_id, "t": g.time_s.to_numpy(float), "I": g.current_a.to_numpy(float),
                        "V": (g["v_charger_cell"] if use_chg else g.voltage_v).to_numpy(float), "T": g.temp_c.to_numpy(float),
                        "v_is_charger": bool(use_chg), "throughput_ah": float(r.throughput_ah),
                        "q_ah": cap_ref * float(np.interp(r.throughput_ah, ct_tp, ct_soh))})
    return out


# ---------------------------------------------------------------- OCV
def fit_ocv_from_reference(segs: List[Dict[str, Any]], p: VoltParams, n_bins: int = 40, n_grid: int = 21) -> Dict[str, Any]:
    """OCV(soc) ~= V - I*(R0(T)+R1), soc from Coulomb counting normalised to each reference discharge (assumes start full, end empty)."""
    S, O = [], []
    for s in segs:
        if s["kind"] != "ref_discharge": continue
        t, I = s["t"], s["I"]; dt = np.diff(t, prepend=t[0])
        cum = np.cumsum(np.abs(I) * dt); soc = 1.0 - cum / cum[-1]
        S.append(soc); O.append(s["V"] - I * (r0_of_T(p.r0, p.r_ea_k, s["T"]) + p.r1))
    if not S: raise ValueError("no reference discharge segments")
    soc, ocv = np.concatenate(S), np.concatenate(O)
    edges = np.linspace(0, 1, n_bins + 1); b = np.clip(np.digitize(soc, edges) - 1, 0, n_bins - 1)
    cen = 0.5 * (edges[1:] + edges[:-1]); med = np.full(n_bins, np.nan); spread = np.full(n_bins, np.nan)
    for k in range(n_bins):
        m = b == k
        if m.sum() >= 5:
            med[k] = np.median(ocv[m]); spread[k] = np.subtract(*np.percentile(ocv[m], [75, 25]))
    ok = ~np.isnan(med)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(cen[ok], med[ok])
    grid = np.linspace(0, 1, n_grid); v = iso.predict(grid)
    lo_c, hi_c = cen[ok].min(), cen[ok].max(); xs = cen[ok]; ys = iso.predict(xs)
    for edge, sel in (("lo", grid < lo_c), ("hi", grid > hi_c)):         # linear extrapolation beyond the data range (slope >= 0)
        if sel.any():
            part = slice(0, 3) if edge == "lo" else slice(-3, None)
            slope = max(np.polyfit(xs[part], ys[part], 1)[0], 0.0)
            ref_x, ref_y = (xs[0], ys[0]) if edge == "lo" else (xs[-1], ys[-1])
            v[sel] = ref_y + slope * (grid[sel] - ref_x)
    v = np.maximum.accumulate(v)
    return {"soc": grid.tolist(), "v": v.tolist(), "median_iqr_V": float(np.nanmedian(spread)), "n_points": int(len(soc))}


# ---------------------------------------------------------------- voltage replay / fit
def infer_soc0(s: Dict[str, Any], p: VoltParams, ocv: OCVCurve) -> float:
    r = float(r0_of_T(p.r0, p.r_ea_k, s["T"][0])) + (p.r_cable if s["v_is_charger"] else 0.0)
    return float(np.clip(ocv.inverse(s["V"][0] - s["I"][0] * r), 0.0, 1.0))


def replay_voltage(s: Dict[str, Any], p: VoltParams, ocv: OCVCurve, soc0: Optional[float] = None) -> np.ndarray:
    t, I, T = s["t"], s["I"], s["T"]; dt = float(np.median(np.diff(t)))
    soc0 = infer_soc0(s, p, ocv) if soc0 is None else soc0
    soc = soc0 + np.cumsum(I) * dt / (3600.0 * s["q_ah"])
    a = np.exp(-dt / p.tau_s); v1 = lfilter([(1 - a) * p.r1], [1, -a], I)
    v = np.interp(soc, ocv.soc, ocv.v) + I * r0_of_T(p.r0, p.r_ea_k, T) + v1
    return v + I * p.r_cable if s["v_is_charger"] else v


def fit_voltage_params(segs: List[Dict[str, Any]], ocv: OCVCurve, p0: VoltParams) -> VoltParams:
    lo = [1e-3, 0.0, 1e-3, TAU_MIN_S, 0.0]; hi = [0.2, 6000.0, 0.2, 900.0, 0.2]
    x0 = np.clip(p0.vec(), np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    def resid(x):
        p = VoltParams(*x)
        return np.concatenate([(replay_voltage(s, p, ocv) - s["V"]) / np.sqrt(len(s["V"])) for s in segs])
    r = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.05, x_scale=[0.01, 1000, 0.01, 50, 0.01])
    return VoltParams(*r.x)


def validate_voltage(segs, p: VoltParams, ocv: OCVCurve) -> pd.DataFrame:
    rows = []
    for s in segs:
        e = replay_voltage(s, p, ocv) - s["V"]
        rows.append({"pack_id": s["pack_id"], "kind": s["kind"], "v_is_charger": s["v_is_charger"], "rmse_v": float(np.sqrt(np.mean(e ** 2))),
                     "mae_v": float(np.mean(np.abs(e))), "bias_v": float(e.mean()), "n": len(e)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- thermal
def replay_temperature(s: Dict[str, Any], p: VoltParams, c_th: float, h: float, t_amb: float) -> np.ndarray:
    t, I, T = s["t"], s["I"], s["T"]; dt = float(np.median(np.diff(t)))
    P = I ** 2 * (r0_of_T(p.r0, p.r_ea_k, T) + p.r1)             # per-cell Joule heat, R(T) from measured T
    a = 1.0 - h * dt / c_th
    theta = lfilter([dt / c_th], [1, -a], P) + (T[0] - t_amb) * a ** np.arange(1, len(T) + 1)
    return t_amb + theta


def fit_thermal(segs, p: VoltParams, t_amb: Dict[str, float], x0=(32.0, 0.04)):
    def resid(x):
        return np.concatenate([(replay_temperature(s, p, x[0], x[1], t_amb[s["pack_id"]]) - s["T"]) / np.sqrt(len(s["T"])) for s in segs])
    r = least_squares(resid, x0, bounds=([10.0, 1e-3], [500.0, 0.5]), loss="soft_l1", f_scale=1.0, x_scale=[10, 0.02])
    return float(r.x[0]), float(r.x[1])


def validate_temperature(segs, p, c_th, h, t_amb) -> pd.DataFrame:
    rows = []
    for s in segs:
        e = replay_temperature(s, p, c_th, h, t_amb[s["pack_id"]]) - s["T"]
        rows.append({"pack_id": s["pack_id"], "kind": s["kind"], "rmse_c": float(np.sqrt(np.mean(e ** 2))), "bias_c": float(e.mean()),
                     "dT_meas_c": float(s["T"][-1] - s["T"][0])})
    return pd.DataFrame(rows)


def ambient_proxy(df_pack: pd.DataFrame) -> float:
    """ASSUMPTION: lab ambient ~ 5th percentile of the pack's battery temperature."""
    return float(np.percentile(df_pack["temp_c"].to_numpy(), 5))


def calibrated_config(cfg: Dict[str, Any], p: VoltParams, ocv: Dict[str, Any], c_th: float, h: float) -> Dict[str, Any]:
    import copy
    c = copy.deepcopy(cfg); s = c["simulator"]
    s.update({"r0_ohm": float(p.r0), "r1_ohm": float(p.r1), "c1_f": float(p.c1), "r_ea_k": float(p.r_ea_k),
              "ocv_soc": ocv["soc"], "ocv_v": ocv["v"]})
    s["thermal"].update({"c_th_j_per_k": c_th, "h_w_per_k": h})
    c["agents"]["nominal_model"].update({"r0_ohm": float(p.r0), "r1_ohm": float(p.r1), "tau_s": float(p.tau_s),
                                         "c_th_j_per_k": c_th, "h_w_per_k": h})
    return c
