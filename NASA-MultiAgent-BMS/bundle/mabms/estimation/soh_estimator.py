"""SOH estimation baselines on the DERIVED cycle table (soh = capacity / own early reference capacity).
Features exclude capacity itself. Normalisation is inside the sklearn Pipeline => fitted on train folds only."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..data.splits import grouped_folds
from ..evaluation.metrics import regression_report, per_group_report

FEATURES = ["w_v0", "w_v_end", "w_dv", "w_i_mean", "w_temp_mean", "w_v_drop_step",
            "v_mean", "i_mean", "temp_mean", "temp_rise"]
# NB: v_mean/i_mean/temp_* use the whole discharge; drop them via `features=` if you want strictly early-window features.


def make_model(name: str, seed: int = 0):
    if name == "mean":
        return DummyRegressor()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "hgb":
        return make_pipeline(StandardScaler(), HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed))
    if name == "xgboost":
        from xgboost import XGBRegressor
        return make_pipeline(StandardScaler(), XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=seed))
    raise ValueError(name)


def grouped_cv(table: pd.DataFrame, model_name: str, n_splits: int, features: List[str] = FEATURES,
               target: str = "soh", seed: int = 0) -> Dict[str, object]:
    X = table[features].to_numpy(float); y = table[target].to_numpy(float); g = table["pack_id"].to_numpy()
    pred = np.full(len(y), np.nan)
    for tr, te in grouped_folds(g, n_splits):
        m = make_model(model_name, seed).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return {"overall": regression_report(y, pred), "per_pack": per_group_report(y, pred, g), "pred": pred}


def fit_final(table: pd.DataFrame, model_name: str, features: List[str] = FEATURES, seed: int = 0):
    return make_model(model_name, seed).fit(table[features].to_numpy(float), table["soh"].to_numpy(float))
