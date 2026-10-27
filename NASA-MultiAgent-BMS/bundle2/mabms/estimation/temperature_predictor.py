"""Future-temperature prediction: persistence vs Ridge vs HistGB on lagged windows (per pack, no cross-pack windows)."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..data.splits import grouped_folds
from ..evaluation.metrics import regression_report, per_group_report


def make_windows(df: pd.DataFrame, lags: int, horizon: int, stride: int, max_gap_factor: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features: [T, V, I] at t and lag differences of T, plus mean I over lags. Target: T(t+horizon) - T(t)."""
    d = df.sort_values("time_s")
    T, V, I = d["temp_c"].to_numpy(), d["voltage_v"].to_numpy(), d["current_a"].to_numpy()
    t = d["time_s"].to_numpy(); med = np.median(np.diff(t)) if len(t) > 1 else 1.0
    idx = np.arange(lags, len(d) - horizon, stride)
    # keep only windows whose time span is ~ (lags+horizon)*median_dt (no segment/time gaps inside the window)
    idx = idx[(t[idx + horizon] - t[idx - lags]) <= max_gap_factor * (lags + horizon) * max(med, 1e-9)]
    X = np.column_stack([T[idx], V[idx], I[idx], np.abs(I[idx]),
                         T[idx] - T[idx - lags],
                         np.array([I[k - lags:k + 1].mean() for k in idx]),
                         np.array([(I[k - lags:k + 1] ** 2).mean() for k in idx])])
    y = T[idx + horizon] - T[idx]
    return X, y, T[idx]


def build(dfs: Dict[str, pd.DataFrame], lags: int, horizon: int, stride: int):
    Xs, ys, T0, G = [], [], [], []
    for pid, d in dfs.items():
        if len(d) <= lags + horizon + 1:
            continue
        X, y, t0 = make_windows(d, lags, horizon, stride)
        Xs.append(X); ys.append(y); T0.append(t0); G.append(np.full(len(y), pid))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(T0), np.concatenate(G)


def grouped_cv(dfs, lags, horizon, stride, n_splits, seed=0) -> Dict[str, object]:
    X, y, T0, g = build(dfs, lags, horizon, stride)
    out = {}
    models = {"persistence": None,
              "ridge": make_pipeline(StandardScaler(), Ridge(1.0)),
              "hgb": make_pipeline(StandardScaler(), HistGradientBoostingRegressor(max_iter=200, random_state=seed))}
    for name, proto in models.items():
        pred = np.zeros(len(y))
        for tr, te in grouped_folds(g, n_splits):
            if proto is None:
                pred[te] = 0.0   # predicts no change (delta target)
            else:
                from sklearn.base import clone
                pred[te] = clone(proto).fit(X[tr], y[tr]).predict(X[te])
        # report error on absolute temperature (T0 + delta)
        out[name] = {"overall_delta": regression_report(y, pred),
                     "overall": regression_report(T0 + y, T0 + pred), "overall_delta": regression_report(y, pred),
                     "per_pack": per_group_report(T0 + y, T0 + pred, g)}
    return out
