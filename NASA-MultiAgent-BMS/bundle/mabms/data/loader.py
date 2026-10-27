"""Load raw files into a canonical schema while preserving provenance."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .cleaning import clean_signals

REQUIRED = ["time", "voltage", "current", "temperature"]


def resolve_columns(df: pd.DataFrame, aliases: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for canon, cands in aliases.items():
        out[canon] = next((c for c in cands if c in df.columns), None)
    missing = [k for k in REQUIRED if out.get(k) is None]
    if missing:
        raise KeyError(f"Could not map required columns {missing}; file columns: {list(df.columns)}. "
                       f"Edit data.columns in configs/default.yaml after reviewing the audit.")
    return out


def pack_id_from_path(path: Path, root: Path, regex: Optional[str]) -> str:
    rel = str(path.relative_to(root).with_suffix(""))
    if regex:
        m = re.search(regex, rel)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return rel.replace("/", "__").replace("\\", "__")


def load_pack_file(path: str | Path, root: str | Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    path, root = Path(path), Path(root)
    dcfg = cfg["data"]
    df = pd.read_csv(path, low_memory=False)
    cm = resolve_columns(df, dcfg["columns"])
    pid = pack_id_from_path(path, root, dcfg["pack_id_regex"])
    regex = dcfg.get("pack_id_regex")
    is_real = bool(regex and re.search(regex, str(path)))
    v_scale = dcfg.get("voltage_scale", 0.5) if is_real else 1.0
    out = pd.DataFrame({
        "time_s": pd.to_numeric(df[cm["time"]], errors="coerce") * dcfg["time_scale_to_s"],
        "voltage_v": pd.to_numeric(df[cm["voltage"]], errors="coerce") * v_scale,
        # canonical convention used everywhere downstream: current > 0 is CHARGE
        "current_a": -dcfg["discharge_current_sign"] * pd.to_numeric(df[cm["current"]], errors="coerce"),
        "temp_c": pd.to_numeric(df[cm["temperature"]], errors="coerce"),
    })
    for opt in ("mode", "mission_type"):
        if cm.get(opt):
            out[opt] = df[cm[opt]]
    out = out.dropna(subset=["time_s", "voltage_v", "current_a", "temp_c"]).reset_index(drop=True)
    out = out[out["voltage_v"] > 0.5].reset_index(drop=True)
    out = clean_signals(out, cfg)
    print(f"[{pid}] clean_signals dropped {out.attrs.get('n_dropped', 0)} samples")
    out["pack_id"] = pid
    out["source_file"] = str(path.relative_to(root))
    return out




def load_all(root: str | Path, cfg: Dict[str, Any], suffixes=(".csv",)) -> Dict[str, pd.DataFrame]:
    """Returns {pack_id: concatenated dataframe sorted by time}. Files sharing a pack_id are concatenated."""
    root = Path(root)
    parts: Dict[str, List[pd.DataFrame]] = {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in suffixes and p.is_file():
            d = load_pack_file(p, root, cfg)
            parts.setdefault(d["pack_id"].iloc[0], []).append(d)
    out = {}
    for k, v in parts.items():
        conc = pd.concat(v, ignore_index=True)
        conc.attrs["n_dropped"] = sum(x.attrs.get("n_dropped", 0) for x in v)
        out[k] = conc
    return out
