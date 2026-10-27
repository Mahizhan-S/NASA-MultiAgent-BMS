"""Shared scalar objective used ONLY for tuning the single-agent baseline and as one summary column."""
from __future__ import annotations
from typing import Any, Dict
import pandas as pd


def objective_J(m: Dict[str, Any], deadline_s: float, cfg: Dict[str, Any]) -> float:
    o = cfg["objective"]
    t = m["time_to_target_s"]
    T = (t / deadline_s) if (t is not None and t == t) else 1.0 + o["fail_penalty"]
    return float(o["w_t"] * T + o["w_d"] * m["capacity_loss"] / o["d_scale"]
                 + o["w_c"] * m["cooling_energy_wh"] / o["e_scale"] + o["w_v"] * m["unsafe_executed_steps"] / o["v_scale"])


def add_J(df: pd.DataFrame, deadlines, cfg) -> pd.DataFrame:
    df = df.copy(); df["J"] = [objective_J(r, deadlines[int(r["scenario_idx"])], cfg) for r in df.to_dict("records")]
    return df


def ensure_scales(cfg: Dict[str, Any], n: int = 20, seed_offset: int = 900) -> Dict[str, float]:
    """Replace objective.d_scale / e_scale given as "auto" (or null) by the MEDIAN (capacity loss) / MEAN, floored at 0.01 Wh (cooling energy) of the conventional baseline on TRAIN-regime scenarios
    (so each J term is measured relative to conventional control). Uses train-regime scenarios only; mutates cfg and returns the scales."""
    o = cfg["objective"]; need = [k for k in ("d_scale", "e_scale") if o.get(k) in (None, "auto")]
    if need:
        import numpy as np
        from ..environment.battery_env import make_scenarios
        from .run_independent import run_benchmark
        df = run_benchmark(cfg, make_scenarios(n, cfg["seed"] + seed_offset, "train"), ["conventional"], base_seed=seed_offset * 100)
        if "d_scale" in need: o["d_scale"] = float(max(np.median(df["capacity_loss"]), 1e-9))
        if "e_scale" in need: o["e_scale"] = float(max(np.mean(df["cooling_energy_wh"]), 0.01))   # mean (median is ~0 when cooling is rarely used), floored
    return {"d_scale": o["d_scale"], "e_scale": o["e_scale"]}
