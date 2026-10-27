"""Estimator comparison. Model selection: grouped CV on dev packs (train+val) ONLY.
Final report: fit on all dev packs, evaluate once on reserved TEST packs. Pairing unit = pack (per-pack MAE)."""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from ..estimation import soh_estimator as soh
from ..estimation import temperature_predictor as tp
from ..estimation.soh_estimator import FEATURES
from .metrics import per_group_report, regression_report
from .stats import bootstrap_ci, holm

EARLY_ONLY = [f for f in FEATURES if f.startswith("w_")]


def _pairwise(per_pack: Dict[str, pd.DataFrame], best_ref: str) -> pd.DataFrame:
    rows = []
    for name, d in per_pack.items():
        if name == best_ref: continue
        m = per_pack[best_ref].merge(d, on="pack_id", suffixes=("_ref", "_alt"))
        diff = (m["mae_alt"] - m["mae_ref"]).to_numpy()          # >0 => reference model better on that pack
        p = 1.0
        if len(diff) >= 2 and np.any(diff != 0):
            try: p = float(stats.wilcoxon(diff).pvalue)
            except ValueError: pass
        lo, hi = bootstrap_ci(diff) if len(diff) >= 2 else (np.nan, np.nan)
        rows.append({"ref": best_ref, "alt": name, "n_packs": len(diff), "mean_mae_increase_alt": float(diff.mean()), "ci95_lo": lo, "ci95_hi": hi,
                     "frac_packs_ref_better": float((diff > 0).mean()), "wilcoxon_p": p})
    t = pd.DataFrame(rows)
    if len(t): t["p_holm"] = holm(t["wilcoxon_p"].tolist())
    return t


def compare_soh(tab: pd.DataFrame, dev: List[str], test: List[str], n_splits: int, models=("mean", "ridge", "hgb"), seed: int = 0) -> Dict[str, Any]:
    dev_t, test_t = tab[tab.pack_id.isin(dev)], tab[tab.pack_id.isin(test)]
    configs = {f"{m}|all": (m, FEATURES) for m in models}
    configs.update({f"{m}|early": (m, EARLY_ONLY) for m in models if m != "mean"})
    if "throughput_ah" in tab.columns:
        configs["ridge|throughput_only"] = ("ridge", ["throughput_ah"])      # naive ageing-axis baseline
    cv_rows, per_pack_cv, test_rows, per_pack_test = [], {}, [], {}
    for name, (m, feats) in configs.items():
        r = soh.grouped_cv(dev_t, m, n_splits, features=feats, seed=seed)
        cv_rows.append({"config": name, **r["overall"]}); per_pack_cv[name] = r["per_pack"]
        if len(test_t):
            mdl = soh.fit_final(dev_t, m, feats, seed); pred = mdl.predict(test_t[feats].to_numpy(float))
            test_rows.append({"config": name, **regression_report(test_t["soh"], pred)})
            per_pack_test[name] = per_group_report(test_t["soh"].to_numpy(), pred, test_t["pack_id"].to_numpy())
    cv = pd.DataFrame(cv_rows).sort_values("mae"); best = cv.iloc[0]["config"]       # best chosen on DEV CV only
    return {"cv": cv, "test": pd.DataFrame(test_rows), "best_by_cv": best, "pairwise_cv": _pairwise(per_pack_cv, best),
            "per_pack_test": per_pack_test}


def compare_temperature(dfs: Dict[str, pd.DataFrame], dev: List[str], test: List[str], T: Dict[str, int], n_splits: int, seed: int = 0) -> Dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = tp.grouped_cv({p: dfs[p] for p in dev if p in dfs}, T["lags"], T["horizon_steps"], T["stride"], n_splits, seed)
    cv_tab = pd.DataFrame([{"model": k, **v["overall"], **{f"delta_{a}": b for a, b in v["overall_delta"].items() if a != "n"}} for k, v in cv.items()]).sort_values("mae")
    out = {"cv": cv_tab, "pairwise_cv": _pairwise({k: v["per_pack"] for k, v in cv.items()}, cv_tab.iloc[0]["model"])}
    tests = [p for p in test if p in dfs]
    if tests:
        Xd, yd, _, _ = tp.build({p: dfs[p] for p in dev if p in dfs}, T["lags"], T["horizon_steps"], T["stride"])
        Xt, yt, T0, gt = tp.build({p: dfs[p] for p in tests}, T["lags"], T["horizon_steps"], T["stride"])
        rows = [{"model": "persistence", **regression_report(T0 + yt, T0), "delta_mae": regression_report(yt, 0 * yt)["mae"], "delta_rmse": regression_report(yt, 0 * yt)["rmse"], "delta_r2": regression_report(yt, 0 * yt)["r2"]}]
        for name, proto in {"ridge": make_pipeline(StandardScaler(), Ridge(1.0)),
                            "hgb": make_pipeline(StandardScaler(), HistGradientBoostingRegressor(max_iter=200, random_state=seed))}.items():
            pr = clone(proto).fit(Xd, yd).predict(Xt); dr = regression_report(yt, pr)
            rows.append({"model": name, **regression_report(T0 + yt, T0 + pr), "delta_mae": dr["mae"], "delta_rmse": dr["rmse"], "delta_r2": dr["r2"]})
        out["test"] = pd.DataFrame(rows)
    return out
