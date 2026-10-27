"""Random-search tuning of the single-agent baseline: select on TRAIN scenarios, confirm top-k on VAL scenarios. Test is never touched."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
from ..agents.single_agent import sample_params
from ..environment.battery_env import make_scenarios
from .objective import add_J
from .run_independent import run_benchmark


def _score(cfg, scenarios, params, base_seed) -> float:
    df = run_benchmark(cfg, scenarios, ["single_agent"], base_seed, policy_params=params)
    return float(add_J(df, {i: s.deadline_s for i, s in enumerate(scenarios)}, cfg)["J"].mean())


def tune_single_agent(cfg: Dict[str, Any], seed: int) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    c = cfg["compare"]; rng = np.random.default_rng(seed)
    tr = make_scenarios(c["tune_scenarios"], seed + 11, "train"); va = make_scenarios(c["tune_scenarios"], seed + 12, "train")
    trials = []
    for k in range(c["tune_samples"]):
        p = sample_params(rng); trials.append({"params": p, "J_train": _score(cfg, tr, p, 5000 + k)})
    top = sorted(trials, key=lambda r: r["J_train"])[:5]
    for r in top: r["J_val"] = _score(cfg, va, r["params"], 7000)
    best = min(top, key=lambda r: r["J_val"])
    return best["params"], trials
