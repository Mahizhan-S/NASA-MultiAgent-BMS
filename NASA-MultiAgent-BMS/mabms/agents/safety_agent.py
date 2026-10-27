"""Safety Agent: deterministic action projection, independent of any learned reward.
Research-simulation component only; NOT a certified BMS. Every intervention is logged.

Hardened Features (Phase G):
1. Temporal consistency and rate-of-change limit (|ΔT| <= 2.0 °C / step).
2. Sensor freeze / stuck detection (identical readings for >= 5 steps under charging).
3. Physics-based thermal observer cross-check (Joule heating accumulation vs reported rise).
4. Deterministic fault-safe fallback (current clamp to conservative limit, forced cooling).
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
from ..estimation.soc_estimator import OCVCurve
from ..limits import current_limit_a
from .base import SafeAction, StateEstimate


class SafetyAgent:
    def __init__(self, cfg: Dict[str, Any], enabled: bool = True):
        self.lim = cfg["limits"]
        self.i_abs = current_limit_a(cfg)
        self.enabled = enabled
        self.v_margin = cfg.get("safety", {}).get("v_margin", 0.0)
        s = cfg["simulator"]
        self.ocv = OCVCurve(s["ocv_soc"], s["ocv_v"])
        self.nm = cfg["agents"]["nominal_model"]
        self.r0 = self.nm["r0_ohm"]
        self.dt = s["dt_s"]
        
        sc = cfg.get("safety", {})
        self.r0_margin = sc.get("r0_margin", 1.0)
        self.mon = {"enabled": True, "thr_c": 4.0, "k": 3, "gain": 0.01, **sc.get("temp_monitor", {})}
        
        # Hardened safety settings
        self.hardened = sc.get("hardened", True)
        self.dt_max_c = sc.get("dt_max_c", 2.0)
        self.max_stuck_steps = sc.get("max_stuck_steps", 5)
        self.fault_derate_c = sc.get("fault_derate_c", 0.5)

        self._Tm = None
        self._Tamb = None
        self._bad = 0
        self._li = 0.0
        self._lc = 0.0
        self._T_prev = None
        self._n_stuck = 0
        self._n_rate_viol = 0
        self._joules_in = 0.0
        self.n_monitor_flags = 0
        self.log: List[Dict[str, Any]] = []
        self.n_calls = 0
        self.n_unsafe_proposed = 0
        self.n_interventions = 0
        self.n_fallbacks = 0

    def reset_stats(self) -> None:
        self.log.clear()
        self.n_calls = self.n_unsafe_proposed = self.n_interventions = self.n_fallbacks = self.n_monitor_flags = 0
        self._Tm = None
        self._Tamb = None
        self._bad = 0
        self._li = 0.0
        self._lc = 0.0
        self._T_prev = None
        self._n_stuck = 0
        self._n_rate_viol = 0
        self._joules_in = 0.0

    @staticmethod
    def _valid(est: StateEstimate) -> bool:
        return (est.finite() and getattr(est, "is_valid", True)
                and 2.0 < est.v_meas < 5.0 and -20 < est.temp_meas < 100 and -0.1 < est.soc < 1.1
                and 0.0 < est.soh <= 1.3)

    def _monitor(self, est: StateEstimate) -> Tuple[float, float, bool, List[str]]:
        """Multi-layer sensor validation combining temporal consistency, stuck detection,
        and open-loop energy balance modeling."""
        m = self.mon
        T = est.temp_meas
        reasons: List[str] = []

        # 1. Temporal Rate-of-Change Consistency Check
        if self.hardened and self._T_prev is not None:
            delta_T = abs(T - self._T_prev)
            if delta_T > self.dt_max_c:
                self._n_rate_viol += 1
                if self._n_rate_viol >= 1:
                    reasons.append("temp_sensor_rate_violation")
            else:
                self._n_rate_viol = max(0, self._n_rate_viol - 1)

        # 2. Stuck-Sensor Invariant Check
        # Under active charging current (>0.8 A), thermal sensor cannot stay bitwise identical for >5 steps
        if self.hardened and self._T_prev is not None:
            if abs(T - self._T_prev) < 1e-4 and self._li > 0.8:
                self._n_stuck += 1
                if self._n_stuck >= self.max_stuck_steps:
                    reasons.append("temp_sensor_stuck")
            else:
                self._n_stuck = max(0, self._n_stuck - 1)

        self._T_prev = T

        if not m["enabled"]:
            flagged = bool(reasons)
            if flagged:
                self.n_monitor_flags += 1
            return T, est.temp_pred, flagged, reasons

        # 3. Model Observer Cross-Check
        if self._Tm is None:
            self._Tm = T
            self._Tamb = T
            self._bad = 0
            self._joules_in = 0.0
            return T, est.temp_pred, False, reasons

        nm = self.nm
        r_tot = nm["r0_ohm"] * max(est.r0_scale, 1.0) + nm["r1_ohm"]
        heat_watts = (self._li ** 2) * r_tot
        self._joules_in += heat_watts * self.dt
        
        # Thermal model update
        self._Tm += self.dt * (heat_watts - nm["h_w_per_k"] * (self._Tm - self._Tamb) - self._lc * nm["cool_max_w"]) / nm["c_th_j_per_k"]
        resid = T - self._Tm

        # Check for model discrepancy
        if abs(resid) > m["thr_c"]:
            self._bad += 1
        else:
            self._bad = max(0, self._bad - 1)

        # Physical Joule-Heating / Negative Bias Invariant:
        # If significant energy (>250 J) has been injected but measured rise is much lower than model, or model is hotter
        if self.hardened and self._joules_in > 250.0:
            if (self._Tm - T) > 2.5:
                self._bad += 2

        flagged = (self._bad >= m["k"]) or bool(reasons)

        if not flagged:
            self._Tm += m["gain"] * resid  # Slow correction so benign drift does not accumulate
        else:
            self.n_monitor_flags += 1
            if "temp_sensor_bias" not in reasons and self._bad >= m["k"]:
                reasons.append("temp_sensor_bias")

        # Guarded temperature reconstruction:
        # On sensor bias/stuck fault, guard against under-reading by using max of sensor, model, or offset
        if flagged:
            fault_offset = 5.0 if any("bias" in r or "stuck" in r for r in reasons) else 0.0
            T_use = max(T, self._Tm, T + fault_offset)
            T_pred_use = max(est.temp_pred, T_use)
        else:
            T_use = T
            T_pred_use = est.temp_pred

        return T_use, T_pred_use, flagged, reasons

    def filter(self, est: StateEstimate, i_prop: float, cool_prop: float) -> SafeAction:
        L = self.lim
        self.n_calls += 1
        reasons: List[str] = []
        i_cap = self.i_abs

        # --- invalid state/action -> deterministic fallback (no charge, full cooling)
        if not (np.isfinite(i_prop) and np.isfinite(cool_prop)) or not self._valid(est):
            self.n_unsafe_proposed += 1
            self.n_interventions += 1
            self.n_fallbacks += 1
            act = SafeAction(0.0, 1.0, True, ["invalid_state_or_action"], True)
            self._li, self._lc = 0.0, 1.0
            self._log(est, i_prop, cool_prop, act)
            return act

        T_meas, T_pred, flagged, fault_reasons = self._monitor(est)
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

        if flagged:
            reasons.extend(fault_reasons)
            # If a severe sensor fault is flagged and temperature is elevated, clamp current
            if T_meas > L["t_derate_start_c"] or T_pred > L["t_derate_start_c"]:
                i_cap = min(i_cap, self.fault_derate_c * est.q_nom_ah)

        # --- projection onto the feasible set
        i = max(i_prop, 0.0)
        if i > i_cap:
            i = i_cap
            reasons.append("current_limit")

        i_v = max((L["v_max"] - self.v_margin - self.ocv.ocv(est.soc) - est.v1) / r0_eff, 0.0)
        if i > i_v:
            i = i_v
            reasons.append("voltage_limit")

        T = max(T_meas, T_pred)
        if T > L["t_derate_start_c"]:
            scale = float(np.clip((L["t_max_c"] - T) / (L["t_max_c"] - L["t_derate_start_c"]), 0.0, 1.0))
            if scale < 1.0:
                i *= scale
                reasons.append("thermal_derate")

        if est.soc >= L["soc_max"]:
            i = 0.0
            reasons.append("soc_limit")

        cool = float(np.clip(cool_prop, 0.0, 1.0))
        if (T > L["t_derate_start_c"] or T_meas > L["t_derate_start_c"]) and cool < 1.0:
            cool = 1.0
            reasons.append("force_cooling")

        intervened = bool(reasons) or (i_prop < 0)
        if intervened:
            self.n_interventions += 1

        act = SafeAction(float(i), cool, intervened, reasons)
        self._li, self._lc = float(i), cool
        if intervened:
            self._log(est, i_prop, cool_prop, act)
        return act

    def _log(self, est, i_prop, cool_prop, act) -> None:
        self.log.append({
            "time_s": est.time_s,
            "soc": est.soc,
            "temp": est.temp_meas,
            "v": est.v_meas,
            "i_prop": i_prop,
            "cool_prop": cool_prop,
            "i_exec": act.current_a,
            "cool_exec": act.cool_cmd,
            "reasons": ";".join(act.reasons),
            "fallback": act.fallback
        })

    def stats(self) -> Dict[str, float]:
        n = max(self.n_calls, 1)
        return {
            "unsafe_proposed": self.n_unsafe_proposed,
            "unsafe_proposed_rate": self.n_unsafe_proposed / n,
            "interventions": self.n_interventions,
            "intervention_rate": self.n_interventions / n,
            "fallbacks": self.n_fallbacks,
            "monitor_flags": self.n_monitor_flags
        }
