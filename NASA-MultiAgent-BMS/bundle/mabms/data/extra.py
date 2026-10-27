"""Attach an extra raw column (e.g. voltage_charger) to an already-loaded pack frame by time, without touching loader.py.
Voltage scaling mirrors the loader: scale only when data.pack_id_regex is set (2S packs -> per-cell-equivalent)."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def attach_extra_column(df_pack: pd.DataFrame, root: str | Path, cfg: Dict[str, Any], aliases: List[str],
                        new_name: str = "v_charger_cell", is_voltage: bool = True) -> pd.DataFrame:
    root = Path(root); d = df_pack.sort_values("time_s").reset_index(drop=True).copy()
    scale = cfg["data"].get("voltage_scale", 1.0) if (is_voltage and cfg["data"].get("pack_id_regex")) else 1.0
    tcands = cfg["data"]["columns"]["time"]; parts = []
    for src in d["source_file"].unique():
        path = root / src
        hdr = pd.read_csv(path, nrows=0).columns
        tcol = next((c for c in tcands if c in hdr), None); ecol = next((c for c in aliases if c in hdr), None)
        if tcol is None or ecol is None:
            continue
        r = pd.read_csv(path, usecols=[tcol, ecol], low_memory=False)
        r = pd.DataFrame({"time_s": pd.to_numeric(r[tcol], errors="coerce") * cfg["data"]["time_scale_to_s"],
                          new_name: pd.to_numeric(r[ecol], errors="coerce") * scale}).dropna().sort_values("time_s")
        parts.append(r)
    if not parts:
        d[new_name] = np.nan
        return d
    r = pd.concat(parts).sort_values("time_s").drop_duplicates("time_s")
    return pd.merge_asof(d, r, on="time_s", tolerance=0.5, direction="nearest")
