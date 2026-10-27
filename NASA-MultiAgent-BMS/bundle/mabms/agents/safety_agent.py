"""Safety Agent: deterministic action projection, independent of any learned reward.
Research-simulation component only; NOT a certified BMS. Every intervention is logged."""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from ..estimation.soc_estimator import OCVCurve
from ..limits import current_limit_a
from .base import SafeAction, StateEstimate


class SafetyAgent:
    def __init__(self, cfg: Dict[str, Any], enabled: bool = True):
        self.lim = cfg["limits"]; self.i_abs = current_limit_a(cfg); self.enabled = enabled; self.v_margin = cfg.get("safety", {}).get("v_margin", 0.0)
        s = cfg["simulator"]; self.ocv = OCVCurve(s["ocv_soc"], s["ocv_v"])
        self.r0 = cfg["agents"]["nominal_model"]["r0_ohm"]
        self.log: List[Dict[str, Any]] = []
        self.n_calls = 0; self.n_unsafe_proposed = 0; self.n_interventions = 0; self.n_fallbacks = 0

    def reset_stats(self) -> None:
        self.log.clear(); self.n_calls = self.n_unsafe_proposed = self.n_interventions = self.n_fallbacks = 0

    @staticmethod
    def _valid(est: StateEstimate) -> bool:
        return (est.finite() and 2.0 < est.v_meas < 5.0 and -20 < est.temp_meas < 100 and -0.1 < est.soc < 1.1
                and 0.0 < est.soh <= 1.3)

    def filter(self, est: StateEstimate, i_prop: float, cool_prop: float) -> SafeAction:
        L = self.lim; self.n_calls += 1
        reasons: List[str] = []
        i_cap = self.i_abs
        # --- invalid state/action -> deterministic fallback (no charge, full cooling)
        if not (np.isfinite(i_prop) and np.isfinite(cool_prop)) or not self._valid(est):
            self.n_unsafe_proposed += 1; self.n_interventions += 1; self.n_fallbacks += 1
            act = SafeAction(0.0, 1.0, True, ["invalid_state_or_action"], True)
            self._log(est, i_prop, cool_prop, act); return act
        # --- detect unsafe proposal (pre-projection)
        v_next = self.ocv.ocv(est.soc) + i_prop * self.r0 + est.v1
        unsafe = []
        if v_next > L["v_max"]: unsafe.append("voltage")
        if i_prop > i_cap + 1e-9: unsafe.append("current")
        if est.temp_pred > L["t_max_c"] and i_prop > 0: unsafe.append("temperature")
        if est.soc >= L["soc_max"] and i_prop > 0: unsafe.append("soc")
        if i_prop < 0: unsafe.append("negative_current")
        if unsafe: self.n_unsafe_proposed += 1
        if not self.enabled:
            return SafeAction(i_prop, float(np.clip(cool_prop, 0, 1)))
        # --- projection onto the feasible set
        i = max(i_prop, 0.0)
        if i > i_cap: i = i_cap; reasons.append("current_limit")
        i_v = max((L["v_max"] - self.v_margin - self.ocv.ocv(est.soc) - est.v1) / self.r0, 0.0)
        if i > i_v: i = i_v; reasons.append("voltage_limit")
        T = max(est.temp_meas, est.temp_pred)
        if T > L["t_derate_start_c"]:
            scale = float(np.clip((L["t_max_c"] - T) / (L["t_max_c"] - L["t_derate_start_c"]), 0.0, 1.0))
            if scale < 1.0: i *= scale; reasons.append("thermal_derate")
        if est.soc >= L["soc_max"]: i = 0.0; reasons.append("soc_limit")
        cool = float(np.clip(cool_prop, 0.0, 1.0))
        if est.temp_meas > L["t_derate_start_c"] and cool < 1.0:
            cool = 1.0; reasons.append("force_cooling")
        intervened = bool(reasons) or (i_prop < 0)
        if intervened: self.n_interventions += 1
        act = SafeAction(float(i), cool, intervened, reasons)
        if intervened: self._log(est, i_prop, cool_prop, act)
        return act

    def _log(self, est, i_prop, cool_prop, act) -> None:
        self.log.append({"time_s": est.time_s, "soc": est.soc, "temp": est.temp_meas, "v": est.v_meas,
                         "i_prop": i_prop, "cool_prop": cool_prop, "i_exec": act.current_a, "cool_exec": act.cool_cmd,
                         "reasons": ";".join(act.reasons), "fallback": act.fallback})

    def stats(self) -> Dict[str, float]:
        n = max(self.n_calls, 1)
        return {"unsafe_proposed": self.n_unsafe_proposed, "unsafe_proposed_rate": self.n_unsafe_proposed / n,
                "interventions": self.n_interventions, "intervention_rate": self.n_interventions / n,
                "fallbacks": self.n_fallbacks}
