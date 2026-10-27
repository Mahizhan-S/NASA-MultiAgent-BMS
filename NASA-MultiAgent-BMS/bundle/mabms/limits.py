"""Single source of truth for the cell current limit. ABSOLUTE amps (datasheet-style), shared by the environment (violation counting,
hardware ceiling), the safety agent and the controllers. Previously the environment scaled the limit with each cell's rated capacity
while the safety agent used nominal capacity x estimated SOH, so the two disagreed on low-capacity cells."""
from __future__ import annotations
from typing import Any, Dict


def current_limit_a(cfg: Dict[str, Any]) -> float:
    lim = cfg["limits"]
    if lim.get("i_max_a") is not None:
        return float(lim["i_max_a"])
    return float(lim["i_max_c"]) * float(cfg["simulator"]["q0_ah"])     # fallback: C-rate x NOMINAL capacity (not per-cell)
