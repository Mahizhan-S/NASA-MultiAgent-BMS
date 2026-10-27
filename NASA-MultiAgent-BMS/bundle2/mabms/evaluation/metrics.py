"""Regression metrics and per-group (per-pack) reporting."""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float("nan") if ss_tot == 0 else float(1 - ss_res / ss_tot)


def regression_report(y, p) -> Dict[str, float]:
    return {"mae": mae(y, p), "rmse": rmse(y, p), "r2": r2(y, p), "n": int(len(y))}


def per_group_report(y, p, groups) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p, "g": groups})
    rows = []
    for g, d in df.groupby("g"):
        rows.append({"pack_id": g, **regression_report(d["y"], d["p"])})
    return pd.DataFrame(rows)
