"""Signal cleaning applied AFTER voltage scaling to per-cell-equivalent. Removes (a) isolated single-sample voltage spikes
(e.g. relay-closing transient at load switch-on), (b) invalid sensor values. Step edges (real load changes) are preserved because
a spike must deviate from BOTH neighbours in the same direction."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd


def isolated_spike_mask(v: np.ndarray, thr: float) -> np.ndarray:
    m = np.zeros(len(v), bool)
    if len(v) < 3:
        return m
    d1, d2 = v[1:-1] - v[:-2], v[1:-1] - v[2:]
    m[1:-1] = (np.abs(d1) > thr) & (np.abs(d2) > thr) & (np.sign(d1) == np.sign(d2))
    return m


def clean_signals(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    c = cfg["data"].get("cleaning", {})
    n0 = len(df)
    df = df.sort_values("time_s").reset_index(drop=True)
    keep = ~isolated_spike_mask(df["voltage_v"].to_numpy(), c.get("spike_dv_cell", 0.1))
    keep &= df["voltage_v"].to_numpy() < c.get("v_cell_hard_max", 4.6)
    for bad in c.get("temp_invalid_values", [0.0]):
        keep &= (df["temp_c"].to_numpy() != bad)
    keep &= df["temp_c"].to_numpy() < c.get("temp_max_valid_c", 150.0)
    out = df[keep].reset_index(drop=True)
    out.attrs["n_dropped"] = n0 - len(out)
    return out
