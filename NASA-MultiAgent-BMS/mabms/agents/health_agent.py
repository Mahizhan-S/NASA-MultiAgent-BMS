"""Health Agent (Phase B: Adaptive degradation-aware health proposer with state-dependent feasibility,
uncertainty handling, noise smoothing, bounded slew-rate, and deterministic safety fallbacks)."""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from ..limits import current_limit_a
from .base import HealthProposal, StateEstimate, Task


class HealthAgent:
    def __init__(self, cfg: Dict[str, Any], mode: str = "adaptive"):
        self.cfg = cfg
        self.c = cfg["agents"]["health"]
        self.lim = cfg["limits"]
        self.soh_eol = cfg["rul"]["soh_eol"]
        self.t_ref = cfg["agents"]["nominal_model"].get("t_ref_c", 25.0)
        self.default_mode = cfg["agents"]["health"].get("mode", mode)
        self.i_abs_max = current_limit_a(cfg)
        self.q0_ah = cfg["simulator"]["q0_ah"]

        # Degradation surrogate parameters for online dLoss prediction
        dg = cfg.get("simulator", {}).get("degradation", {})
        self.k_deg = float(dg.get("k_deg", 1.0e-4))
        self.ea_deg_k = float(dg.get("ea_deg_k", 3000.0))
        self.c_rate_coeff = float(dg.get("c_rate_coeff", 0.5))
        self.soc_stress = float(dg.get("soc_stress", 1.0))

        # Slew rate and filter parameters
        self.slew_rate_c_per_s = float(self.c.get("slew_rate_c_per_s", 0.5))
        self.ema_alpha_r0 = float(self.c.get("ema_alpha_r0", 0.20))
        self.ema_alpha_temp = float(self.c.get("ema_alpha_temp", 0.20))

        # Filter states
        self._r0_filtered: Optional[float] = None
        self._temp_filtered: Optional[float] = None
        self._last_c_prop: Optional[float] = None
        self._last_time_s: Optional[float] = None

    def reset(self) -> None:
        """Reset internal filter states at the beginning of an episode."""
        self._r0_filtered = None
        self._temp_filtered = None
        self._last_c_prop = None
        self._last_time_s = None

    def _validate_inputs(self, est: StateEstimate, task: Task) -> bool:
        """Check for NaN/Inf, physical domain boundaries, and valid task parameters."""
        if not est.finite():
            return False
        if not (-0.1 <= est.soc <= 1.2):
            return False
        if not (0.0 < est.soh <= 1.5):
            return False
        if not (1.0 <= est.v_meas <= 5.5):
            return False
        if not (-50.0 <= est.temp_meas <= 120.0):
            return False
        if not (np.isfinite(task.deadline_s) and np.isfinite(task.target_soc)):
            return False
        if task.deadline_s < 0.0 or task.target_soc <= 0.0:
            return False
        return True

    def _predict_dloss(self, c_rate: float, soc: float, temp_c: float, dt_s: float = 1.0) -> float:
        """Evaluate predicted incremental capacity loss dLoss over dt_s."""
        k0 = 273.15
        f_t = float(np.exp(self.ea_deg_k * (1.0 / (self.t_ref + k0) - 1.0 / (temp_c + k0))))
        f_c = 1.0 + self.c_rate_coeff * max(c_rate - 0.5, 0.0)
        f_s = 1.0 + self.soc_stress * (max(soc - 0.8, 0.0) / 0.2) ** 2
        ah_throughput = (c_rate * self.q0_ah * dt_s) / 3600.0
        return float(self.k_deg * (ah_throughput / self.q0_ah) * f_t * f_c * f_s)

    def propose(self, est: StateEstimate, task: Task, mode: Optional[str] = None) -> HealthProposal:
        use_mode = mode or self.default_mode
        c = self.c
        min_c = float(c.get("min_c", 0.3))
        max_c_hard = float(self.lim.get("i_max_c", 2.45))

        # 1. Deterministic Fallback on Invalid Inputs
        if not self._validate_inputs(est, task):
            v_cv_fallback = float(self.lim["v_max"] - 0.10)
            return HealthProposal(
                max_c_rate=min_c,
                v_cv=v_cv_fallback,
                note="fallback:invalid_state_or_task",
                feasibility_status="INFEASIBLE",
                predicted_dloss=0.0,
                i_required_a=0.0,
                is_fallback=True
            )

        # 2. Timing and Charge Demand
        time_left = task.deadline_s - est.time_s
        remaining_soc = max(task.target_soc - est.soc, 0.0)
        efficiency = float(np.clip(c.get("efficiency", 0.98), 0.5, 1.0))

        # Handle zero remaining time / already-finished charging
        if remaining_soc <= 1e-5:
            # Target reached; zero required current
            i_required = 0.0
            required_c = 0.0
            deadline_c = min_c
            feasibility_status = "FEASIBLE"
        elif time_left <= 0.0:
            # Deadline expired before reaching target
            i_required = float("inf")
            required_c = max_c_hard
            deadline_c = max_c_hard
            feasibility_status = "INFEASIBLE"
        else:
            time_left_eff = max(time_left, 1.0)
            # Standard first-order required average current (in Amperes)
            i_required = (3600.0 * self.q0_ah * remaining_soc) / (efficiency * time_left_eff)
            required_c = i_required / self.q0_ah
            deadline_c = required_c * float(c.get("deadline_margin", 1.15))
            feasibility_status = "FEASIBLE" if i_required <= self.i_abs_max else "INFEASIBLE"

        # 3. Static Mode Baseline (for exact historical reproducibility)
        if use_mode == "static":
            health_scale = 0.5 + 0.5 * float(np.clip((est.soh - self.soh_eol) / max(1.0 - self.soh_eol, 1e-3), 0.0, 1.0))
            cap = c["base_c_max"] * health_scale
            if est.temp_pred > c.get("t_soft_c", 38.0):
                cap *= 0.7
            max_c = float(np.clip(deadline_c, min_c, cap))
            slack = time_left / max(remaining_soc * 3600.0 / max(cap, 1e-6), 1.0)
            v_cv = float(self.lim["v_max"] - (c.get("dv_health", 0.05) if slack > 1.5 else 0.0))
            dloss_pred = self._predict_dloss(max_c, est.soc, est.temp_meas)
            return HealthProposal(
                max_c_rate=max_c,
                v_cv=v_cv,
                note=f"mode=static,req_c={required_c:.2f},cap={cap:.2f},slack={slack:.2f}",
                feasibility_status=feasibility_status,
                predicted_dloss=dloss_pred,
                i_required_a=float(np.nan_to_num(i_required, posinf=self.i_abs_max)),
                is_fallback=False
            )

        # 4. Phase B Adaptive Health Mode
        # (a) Exponential Moving Average Smoothing on Resistance and Temperature
        raw_r0 = max(getattr(est, "r0_scale", 1.0), 1.0)
        if self._r0_filtered is None:
            self._r0_filtered = raw_r0
        else:
            self._r0_filtered = (1.0 - self.ema_alpha_r0) * self._r0_filtered + self.ema_alpha_r0 * raw_r0
        r0_scale = self._r0_filtered

        temp_val = est.temp_pred if np.isfinite(est.temp_pred) else est.temp_meas
        if self._temp_filtered is None:
            self._temp_filtered = temp_val
        else:
            self._temp_filtered = (1.0 - self.ema_alpha_temp) * self._temp_filtered + self.ema_alpha_temp * temp_val
        temp_smooth = self._temp_filtered

        # (b) Health Uncertainty Handling (lower confidence bound)
        soh_sigma = max(getattr(est, "soh_sigma", 0.02), 0.0)
        soh_conservative = max(est.soh - 1.96 * soh_sigma, 0.01)
        soh_norm = float(np.clip((soh_conservative - self.soh_eol) / max(1.0 - self.soh_eol, 1e-3), 0.0, 1.0))
        f_soh = 0.6 + 0.4 * soh_norm

        # (c) Resistance factor (ablation: disable if mode == 'adaptive_no_resistance')
        if use_mode == "adaptive_no_resistance":
            f_rho = 1.0
        else:
            f_rho = 1.0 / np.sqrt(r0_scale)

        # (d) Temperature factor (ablation: disable if mode == 'adaptive_no_degradation_prediction')
        t_soft = float(c.get("t_soft_c", 38.0))
        if use_mode == "adaptive_no_degradation_prediction":
            f_temp = 1.0
            f_soc = 1.0
        else:
            f_temp = float(np.exp(-0.08 * (temp_smooth - t_soft))) if temp_smooth > t_soft else 1.0
            f_soc = 1.0 - 0.35 * (((est.soc - 0.8) / 0.2) ** 2) if est.soc > 0.8 else 1.0

        # Health degradation ceiling (maximum current consistent with gentle aging)
        c_health_max = float(c["base_c_max"] * f_soh * f_rho * f_temp * f_soc)
        i_health_max = c_health_max * self.q0_ah * est.soh

        # Update Feasibility Status:
        if feasibility_status != "INFEASIBLE":
            if i_required > i_health_max:
                feasibility_status = "AT_RISK"
            else:
                feasibility_status = "FEASIBLE"

        # (f) Deadline-Health Reconciliation with Hardware Limits
        # If deadline is urgent (deadline_c > c_health_max), expand towards deadline floor
        # up to physical hardware limits; otherwise enforce gentle health ceiling.
        # (ablation: disable deadline expansion if mode == 'adaptive_no_deadline_feedback')
        if use_mode == "adaptive_no_deadline_feedback":
            c_target = c_health_max
        else:
            c_target = max(c_health_max, deadline_c)
        c_bounded = float(np.clip(c_target, min_c, max_c_hard))

        # (g) Bounded Current Slew-Rate Limiter
        if self._last_c_prop is not None and self._last_time_s is not None:
            dt = max(est.time_s - self._last_time_s, 0.0)
            max_delta_c = self.slew_rate_c_per_s * max(dt, 1.0)
            max_c = float(np.clip(c_bounded, self._last_c_prop - max_delta_c, self._last_c_prop + max_delta_c))
        else:
            max_c = c_bounded

        self._last_c_prop = max_c
        self._last_time_s = est.time_s

        # (h) CV Voltage Reduction
        slack = time_left / max(remaining_soc * 3600.0 / max(c_health_max, 1e-6), 1.0) if time_left > 0 else 0.0
        v_cv = float(self.lim["v_max"] - (c.get("dv_health", 0.05) if slack > 1.3 else 0.0))

        # (i) Online Predicted Incremental Degradation
        dloss_pred = self._predict_dloss(max_c, est.soc, temp_smooth)

        note = (f"mode=adaptive,status={feasibility_status},req_c={required_c:.2f},"
                f"c_health_max={c_health_max:.2f},f_rho={f_rho:.2f},slack={slack:.2f}")

        return HealthProposal(
            max_c_rate=max_c,
            v_cv=v_cv,
            note=note,
            feasibility_status=feasibility_status,
            predicted_dloss=dloss_pred,
            i_required_a=float(np.nan_to_num(i_required, posinf=self.i_abs_max)),
            is_fallback=False
        )
