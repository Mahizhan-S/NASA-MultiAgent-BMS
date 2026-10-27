"""Empirical degradation law fitted to REFERENCE-SOH trajectories, same functional form as the simulator:
   dloss = k * (|I| dt/3600/q0) * f_T * f_C * f_S ,  f_T=exp(Ea*(1/Tref-1/T)),  f_C=1+a*max(C-0.5,0),  f_S=1+b*((max(soc-0.8,0)/0.2)^2)
integrated over the MEASURED traces between consecutive reference checks. Fit to cumulative SOH loss at checks (less noisy than differences).
Model selection by held-out (validation) packs; CIs by pack bootstrap. Identifiability is limited: conditions are confounded with packs."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from .calibration import K0, VoltParams, r0_of_T
from ..estimation.soc_estimator import OCVCurve

VARIANTS = {"k_only": (0, 0, 0), "k_T": (1, 0, 0), "k_T_C": (1, 1, 0), "k_T_C_S": (1, 1, 1)}
T_REF = 25.0


def prepare_pack(df_pack: pd.DataFrame, ct: pd.DataFrame, cfg: Dict[str, Any], ocv: OCVCurve, vp: VoltParams,
                 q0_ah: float, stride: int = 10, i_thr: float = 0.05) -> Dict[str, Any]:
    d = df_pack.sort_values("time_s").reset_index(drop=True)
    t = d.time_s.to_numpy(float)[::stride]; I = d.current_a.to_numpy(float)[::stride]
    V = d.voltage_v.to_numpy(float)[::stride]; T = d.temp_c.to_numpy(float)[::stride]
    dt = np.diff(t, append=t[-1]); dt = np.where(dt > cfg["data"]["segmentation"]["max_gap_s"], 0.0, dt)
    idx = np.searchsorted(ct["t_start"].to_numpy(float), t, side="right") - 1
    valid = (np.abs(I) > i_thr) & (idx >= 0) & (idx < len(ct) - 1)
    soc = np.interp(V - I * (r0_of_T(vp.r0, vp.r_ea_k, T) + vp.r1), ocv.v, ocv.soc)
    return {"ah_norm": (np.abs(I) * dt / 3600.0 / q0_ah)[valid], "T_c": T[valid], "crate": (np.abs(I) / q0_ah)[valid],
            "s": ((np.maximum(soc - 0.8, 0.0) / 0.2) ** 2)[valid], "idx": idx[valid], "n_int": len(ct) - 1,
            "soh": ct["soh"].to_numpy(float), "throughput_ah": ct["throughput_ah"].to_numpy(float)}


def predict_cum_loss(pk: Dict[str, Any], k: float, ea: float, a: float, b: float) -> np.ndarray:
    fT = np.exp(ea * (1.0 / (T_REF + K0) - 1.0 / (pk["T_c"] + K0)))
    fC = 1.0 + a * np.maximum(pk["crate"] - 0.5, 0.0); fS = 1.0 + b * pk["s"]
    per_int = np.bincount(pk["idx"], weights=k * pk["ah_norm"] * fT * fC * fS, minlength=pk["n_int"])
    return np.cumsum(per_int)


def observed_cum_loss(pk) -> np.ndarray:
    return pk["soh"][0] - pk["soh"][1:]


def _unpack(x, mask):
    k = float(np.exp(x[0])); vals = [0.0, 0.0, 0.0]; j = 1
    for i in range(3):
        if mask[i]: vals[i] = x[j]; j += 1
    return k, vals[0], vals[1], vals[2]


def fit_variant(packs: List[Dict[str, Any]], variant: str, k0: Optional[float] = None) -> Dict[str, float]:
    mask = VARIANTS[variant]
    if k0 is None:
        k0 = sum(observed_cum_loss(p)[-1] for p in packs) / max(sum(p["ah_norm"].sum() for p in packs), 1e-9)
    x0 = [np.log(max(k0, 1e-7))]; lo = [np.log(1e-7)]; hi = [np.log(1e-1)]
    for flag, (v0, l, h) in zip(mask, [(2000.0, 0.0, 12000.0), (0.3, 0.0, 5.0), (1.0, 0.0, 20.0)]):
        if flag: x0.append(v0); lo.append(l); hi.append(h)
    def resid(x):
        k, ea, a, b = _unpack(x, mask)
        return np.concatenate([(predict_cum_loss(p, k, ea, a, b) - observed_cum_loss(p)) / np.sqrt(len(p["soh"]) - 1) for p in packs])
    r = least_squares(resid, x0, bounds=(lo, hi), x_scale=[1.0] + [v for f, v in zip(mask, [1000.0, 0.3, 1.0]) if f])
    k, ea, a, b = _unpack(r.x, mask)
    return {"k": k, "ea": ea, "a": a, "b": b}


def evaluate(packs: List[Dict[str, Any]], pid: List[str], par: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for name, p in zip(pid, packs):
        pred = p["soh"][0] - predict_cum_loss(p, par["k"], par["ea"], par["a"], par["b"]); e = pred - p["soh"][1:]
        rows.append({"pack_id": name, "mae_soh": float(np.mean(np.abs(e))), "rmse_soh": float(np.sqrt(np.mean(e ** 2))),
                     "end_err_soh": float(e[-1]), "n_checks": len(e)})
    return pd.DataFrame(rows)


def bootstrap_params(packs: List[Dict[str, Any]], variant: str, n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed); rows = []
    for _ in range(n):
        sel = rng.integers(0, len(packs), len(packs))
        try: rows.append(fit_variant([packs[i] for i in sel], variant))
        except Exception: pass
    return pd.DataFrame(rows)
