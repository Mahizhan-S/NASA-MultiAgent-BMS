"""RUL (remaining cycles) with an EXPLICIT end-of-life criterion (SOH <= soh_eol on the derived SOH).
Only packs whose trajectory actually reaches EOL get labels; others are reported as censored and excluded."""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..data.splits import grouped_folds
from ..evaluation.metrics import regression_report, per_group_report


def build_rul_table(cycle_tables: Dict[str, pd.DataFrame], soh_eol: float, min_cycles: int, slope_k: int = 5):
    rows, censored = [], []
    for pid, t in cycle_tables.items():
        t = t.sort_values("cycle_idx").reset_index(drop=True)
        below = np.where(t["soh"].to_numpy() <= soh_eol)[0]
        if len(t) < min_cycles or len(below) == 0:
            censored.append({"pack_id": pid, "n_cycles": len(t), "min_soh": float(t["soh"].min()) if len(t) else np.nan})
            continue
        eol = int(t["cycle_idx"].iloc[below[0]])
        soh = t["soh"].to_numpy()
        for k in range(slope_k, below[0]):
            sl = np.polyfit(np.arange(slope_k + 1), soh[k - slope_k:k + 1], 1)[0]
            rows.append({"pack_id": pid, "cycle_idx": int(t["cycle_idx"].iloc[k]), "soh": soh[k], "soh_slope": sl,
                         "rul_cycles": eol - int(t["cycle_idx"].iloc[k])})
    return pd.DataFrame(rows), pd.DataFrame(censored)


def grouped_cv(tbl: pd.DataFrame, n_splits: int, seed: int = 0) -> Dict[str, object]:
    feats = ["cycle_idx", "soh", "soh_slope"]
    X, y, g = tbl[feats].to_numpy(float), tbl["rul_cycles"].to_numpy(float), tbl["pack_id"].to_numpy()
    out = {}
    from sklearn.base import clone
    for name, proto in {"ridge": make_pipeline(StandardScaler(), Ridge(1.0)),
                        "hgb": make_pipeline(StandardScaler(), HistGradientBoostingRegressor(max_iter=200, random_state=seed))}.items():
        pred = np.zeros(len(y))
        for tr, te in grouped_folds(g, n_splits):
            pred[te] = clone(proto).fit(X[tr], y[tr]).predict(X[te])
        out[name] = {"overall": regression_report(y, pred), "per_pack": per_group_report(y, pred, g)}
    return out


# =====================  v2: throughput-based RUL with fair baselines and censoring-aware reporting  =====================
def _eol_throughput(t: pd.DataFrame, soh_eol: float):
    """Throughput (Ah) at first reference check with soh<=eol (linear interpolation from the previous check); None if censored."""
    t = t.sort_values("throughput_ah").reset_index(drop=True)
    hit = np.where(t["soh"].to_numpy() <= soh_eol)[0]
    if len(hit) == 0:
        return None
    k = int(hit[0])
    if k == 0:
        return float(t["throughput_ah"].iloc[0])
    x0, x1, y0, y1 = t["throughput_ah"].iloc[k - 1], t["throughput_ah"].iloc[k], t["soh"].iloc[k - 1], t["soh"].iloc[k]
    return float(x0 + (soh_eol - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)


def _history_features(t: pd.DataFrame, k: int, soh_eol: float, slope_k: int) -> Dict[str, float]:
    """Features/baselines using ONLY reference checks 0..k (no future information)."""
    x, y = t["throughput_ah"].to_numpy()[: k + 1], t["soh"].to_numpy()[: k + 1]
    f = {"throughput_ah": float(x[-1]), "soh": float(y[-1]), "soh_slope": np.nan, "global_lin_rul": np.nan}
    if k >= 1:
        lo = max(0, k - slope_k)
        f["soh_slope"] = float(np.polyfit(x[lo:], y[lo:], 1)[0]) if x[-1] > x[lo] else np.nan
        if x[-1] > x[0]:
            b, a = np.polyfit(x, y, 1)                      # global fit from the first reference check
            if b < 0:
                f["global_lin_rul"] = float(max((soh_eol - a) / b - x[-1], 0.0))
    return f


def build_rul_table_v2(cycle_tables: Dict[str, pd.DataFrame], soh_eol: float, min_refs: int = 6, slope_k: int = 3):
    rows, cens = [], []
    for pid, t in cycle_tables.items():
        t = t.sort_values("throughput_ah").reset_index(drop=True)
        if len(t) < min_refs:
            cens.append({"pack_id": pid, "status": "too_few_refs", "n_refs": len(t)}); continue
        eol = _eol_throughput(t, soh_eol)
        last_tp = float(t["throughput_ah"].iloc[-1])
        if eol is None:
            cens.append({"pack_id": pid, "status": "censored", "n_refs": len(t), "min_soh": float(t["soh"].min()), "last_throughput_ah": last_tp}); continue
        for k in range(2, len(t)):
            if t["soh"].iloc[k] <= soh_eol or t["throughput_ah"].iloc[k] >= eol:
                break
            rows.append({"pack_id": pid, "ref_idx": k, "eol_throughput_ah": eol,
                         "rul_ah": eol - float(t["throughput_ah"].iloc[k]), **_history_features(t, k, soh_eol, slope_k)})
    return pd.DataFrame(rows), pd.DataFrame(cens)


def _fit_predict(name: str, tr: pd.DataFrame, te: pd.DataFrame, seed: int) -> np.ndarray:
    if name == "global_linear":
        return te["global_lin_rul"].fillna(tr["rul_ah"].mean()).to_numpy()    # fallback when slope>=0: train-mean RUL
    if name == "mean_eol_lookup":
        return np.maximum(tr.groupby("pack_id")["eol_throughput_ah"].first().mean() - te["throughput_ah"].to_numpy(), 0.0)
    feats = ["throughput_ah", "soh", "soh_slope"]
    Xtr, Xte = tr[feats].fillna(0.0).to_numpy(float), te[feats].fillna(0.0).to_numpy(float)
    from sklearn.base import clone
    proto = make_pipeline(StandardScaler(), Ridge(1.0)) if name == "ridge" else make_pipeline(StandardScaler(), HistGradientBoostingRegressor(max_iter=200, random_state=seed))
    return clone(proto).fit(Xtr, tr["rul_ah"].to_numpy(float)).predict(Xte)


MODELS_V2 = ["global_linear", "mean_eol_lookup", "ridge", "hgb"]


def rul_study_v2(tbl: pd.DataFrame, cycle_tables: Dict[str, pd.DataFrame], censored: pd.DataFrame, soh_eol: float,
                 n_splits: int, seed: int = 0, slope_k: int = 3, censor_frac: float = 0.5) -> Dict[str, object]:
    """Leave-pack-group-out CV on EOL packs; separate censored-pack consistency check (lower-bound violations)."""
    g = tbl["pack_id"].to_numpy(); y = tbl["rul_ah"].to_numpy(float)
    res, per_pack = {}, {}
    for name in MODELS_V2:
        pred = np.zeros(len(y))
        for tr, te in grouped_folds(g, n_splits):
            pred[te] = _fit_predict(name, tbl.iloc[tr], tbl.iloc[te], seed)
        res[name] = regression_report(y, pred); per_pack[name] = per_group_report(y, pred, g)
    out = {"eol_packs": pd.DataFrame([{"model": k, **v} for k, v in res.items()]).sort_values("mae"), "per_pack": per_pack}
    # censored packs: at `censor_frac` of their history the predicted EOL must not fall below their observed lifetime lower bound
    crow = []
    for _, r in censored[censored.get("status", "") == "censored"].iterrows() if len(censored) else []:
        t = cycle_tables[r["pack_id"]].sort_values("throughput_ah").reset_index(drop=True)
        k = max(2, int(censor_frac * (len(t) - 1)))
        feat = pd.DataFrame([_history_features(t, k, soh_eol, slope_k)])
        for name in MODELS_V2:
            rul_hat = float(_fit_predict(name, tbl, feat, seed)[0])
            pred_eol = float(feat["throughput_ah"].iloc[0]) + rul_hat
            crow.append({"pack_id": r["pack_id"], "model": name, "pred_eol_ah": pred_eol, "observed_lower_bound_ah": float(r["last_throughput_ah"]),
                         "violates_lower_bound": pred_eol < float(r["last_throughput_ah"])})
    out["censored_check"] = pd.DataFrame(crow)
    return out
