"""Paired statistics for method comparisons. Pairing unit = scenario (or pack); never pool seeds as independent samples."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    means = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(run, 1.0)
    return adj.tolist()


def paired_compare(df: pd.DataFrame, metric: str, a: str, b: str, pair_key: str = "scenario_idx", method_col: str = "method") -> Dict:
    """Mean over seeds within each pair unit first, then paired tests on the per-unit means. diff = a - b."""
    g = df.groupby([pair_key, method_col])[metric].mean().unstack()
    d = (g[a] - g[b]).dropna().to_numpy(float)
    out = {"metric": metric, "a": a, "b": b, "n_pairs": int(len(d)), "mean_a": float(g[a].mean()), "mean_b": float(g[b].mean()),
           "mean_diff": float(d.mean()) if len(d) else np.nan}
    if len(d) >= 2:
        out["ci95_lo"], out["ci95_hi"] = bootstrap_ci(d)
        out["cohens_dz"] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            out["wilcoxon_p"] = 1.0
        out["win_rate_a_lower"] = float((d < 0).mean())
    return out


def compare_table(df: pd.DataFrame, metrics: List[str], baseline: str, others: List[str], pair_key="scenario_idx") -> pd.DataFrame:
    rows = [paired_compare(df, m, o, baseline, pair_key) for m in metrics for o in others]
    t = pd.DataFrame(rows)
    if "wilcoxon_p" in t:
        t["p_holm"] = holm(t["wilcoxon_p"].fillna(1.0).tolist())
    return t


def summarize(df: pd.DataFrame, metrics: List[str], method_col="method") -> pd.DataFrame:
    return df.groupby(method_col)[metrics].agg(["mean", "std"]).round(5)
