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
        self.nm = cfg["agents"]["nominal_model"]; self.r0 = self.nm["r0_ohm"]; self.dt = s["dt_s"]
        sc = cfg.get("safety", {}); self.r0_margin = sc.get("r0_margin", 1.0); self.mon = {"enabled": False, "thr_c": 4.0, "k": 3, "gain": 0.01, **sc.get("temp_monitor", {})}
        self._Tm = None; self._Tamb = None; self._bad = 0; self._li = 0.0; self._lc = 0.0; self.n_monitor_flags = 0
        self.log: List[Dict[str, Any]] = []
        self.n_calls = 0; self.n_unsafe_proposed = 0; self.n_interventions = 0; self.n_fallbacks = 0

    def reset_stats(self) -> None:
        self.log.clear(); self.n_calls = self.n_unsafe_proposed = self.n_interventions = self.n_fallbacks = self.n_monitor_flags = 0
        self._Tm = None; self._Tamb = None; self._bad = 0; self._li = 0.0; self._lc = 0.0

    @staticmethod
    def _valid(est: StateEstimate) -> bool:
        return (est.finite() and 2.0 < est.v_meas < 5.0 and -20 < est.temp_meas < 100 and -0.1 < est.soc < 1.1
                and 0.0 < est.soh <= 1.3)

    def _monitor(self, est: StateEstimate):
        """Open-loop thermal model (nominal params, ambient := first reading) cross-checks the temperature sensor. If the sensor disagrees with the
        model by > thr for k consecutive steps, the safety logic uses max(sensor, model). Catches stuck / biased sensors the controller shares."""
        m = self.mon
        if not m["enabled"]: return est.temp_meas, est.temp_pred, False
        T = est.temp_meas
        if self._Tm is None:
            self._Tm = T; self._Tamb = T; self._bad = 0; return T, est.temp_pred, False
        nm = self.nm; heat = self._li ** 2 * (nm["r0_ohm"] * max(est.r0_scale, 1.0) + nm["r1_ohm"])
        self._Tm += self.dt * (heat - nm["h_w_per_k"] * (self._Tm - self._Tamb) - self._lc * nm["cool_max_w"]) / nm["c_th_j_per_k"]
        resid = T - self._Tm; self._bad = self._bad + 1 if abs(resid) > m["thr_c"] else 0
        flagged = self._bad >= m["k"]
        if not flagged: self._Tm += m["gain"] * resid          # slow correction so model drift does not accumulate
        else: self.n_monitor_flags += 1
        T_use = max(T, self._Tm) if flagged else T
        return T_use, (max(est.temp_pred, T_use) if flagged else est.temp_pred), flagged

    def filter(self, est: StateEstimate, i_prop: float, cool_prop: float) -> SafeAction:
        L = self.lim; self.n_calls += 1
        reasons: List[str] = []
        i_cap = self.i_abs
        # --- invalid state/action -> deterministic fallback (no charge, full cooling)
        if not (np.isfinite(i_prop) and np.isfinite(cool_prop)) or not self._valid(est):
            self.n_unsafe_proposed += 1; self.n_interventions += 1; self.n_fallbacks += 1
            act = SafeAction(0.0, 1.0, True, ["invalid_state_or_action"], True)
            self._li, self._lc = 0.0, 1.0
            self._log(est, i_prop, cool_prop, act); return act
        T_meas, T_pred, flagged = self._monitor(est)
        r0_eff = self.r0 * max(est.r0_scale, 1.0) * self.r0_margin
        # --- detect unsafe proposal (pre-projection)
        v_next = self.ocv.ocv(est.soc) + i_prop * r0_eff + est.v1
        unsafe = []
        if v_next > L["v_max"]: unsafe.append("voltage")
        if i_prop > i_cap + 1e-9: unsafe.append("current")
        if T_pred > L["t_max_c"] and i_prop > 0: unsafe.append("temperature")
        if est.soc >= L["soc_max"] and i_prop > 0: unsafe.append("soc")
        if i_prop < 0: unsafe.append("negative_current")
        if unsafe: self.n_unsafe_proposed += 1
        if not self.enabled:
            return SafeAction(i_prop, float(np.clip(cool_prop, 0, 1)))
        if flagged: reasons.append("temp_sensor_implausible")
        # --- projection onto the feasible set
        i = max(i_prop, 0.0)
        if i > i_cap: i = i_cap; reasons.append("current_limit")
        i_v = max((L["v_max"] - self.v_margin - self.ocv.ocv(est.soc) - est.v1) / r0_eff, 0.0)
        if i > i_v: i = i_v; reasons.append("voltage_limit")
        T = max(T_meas, T_pred)
        if T > L["t_derate_start_c"]:
            scale = float(np.clip((L["t_max_c"] - T) / (L["t_max_c"] - L["t_derate_start_c"]), 0.0, 1.0))
            if scale < 1.0: i *= scale; reasons.append("thermal_derate")
        if est.soc >= L["soc_max"]: i = 0.0; reasons.append("soc_limit")
        cool = float(np.clip(cool_prop, 0.0, 1.0))
        if T_meas > L["t_derate_start_c"] and cool < 1.0:
            cool = 1.0; reasons.append("force_cooling")
        intervened = bool(reasons) or (i_prop < 0)
        if intervened: self.n_interventions += 1
        act = SafeAction(float(i), cool, intervened, reasons)
        self._li, self._lc = float(i), cool
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
                "fallbacks": self.n_fallbacks, "monitor_flags": self.n_monitor_flags}
