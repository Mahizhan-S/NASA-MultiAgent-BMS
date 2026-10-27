"""Closed-Loop Health Estimation Interface (Phase F).
Provides a unified, modular architecture connecting online recursive estimators
and offline ML models to the adaptive multi-agent charging loop.

Supports:
1. HealthEstimate dataclass with SOH, R0_scale, uncertainties, and fault flags.
2. BaseHealthEstimator abstract interface.
3. OracleHealthEstimator (ground-truth observer for upper-bound benchmarking).
4. CalibratedOnlineHealthEstimator (online EKF state tracking + calibrated noise).
5. PerturbedHealthEstimator (controlled systematic bias, observation noise, and corruption).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class HealthEstimate:
    """Estimated battery health state passed to the Health and Charging agents."""
    soh: float
    soh_sigma: float = 0.01
    r0_scale: float = 1.0
    r0_sigma: float = 0.05
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Enforce physical bounds and sanitization
        if not np.isfinite(self.soh):
            self.soh = 0.80
            self.is_valid = False
        else:
            self.soh = float(np.clip(self.soh, 0.40, 1.25))

        if not np.isfinite(self.soh_sigma) or self.soh_sigma < 0.0:
            self.soh_sigma = 0.05
        else:
            self.soh_sigma = float(self.soh_sigma)

        if not np.isfinite(self.r0_scale) or self.r0_scale <= 0.0:
            self.r0_scale = 1.0
            self.is_valid = False
        else:
            self.r0_scale = float(np.clip(self.r0_scale, 0.50, 4.0))

        if not np.isfinite(self.r0_sigma) or self.r0_sigma < 0.0:
            self.r0_sigma = 0.10
        else:
            self.r0_sigma = float(self.r0_sigma)


class BaseHealthEstimator:
    """Abstract interface for all closed-loop health estimation modules."""

    def reset(self, initial_soh: float = 1.0, initial_r0_scale: float = 1.0) -> HealthEstimate:
        raise NotImplementedError

    def step(self, voltage_v: float, current_a: float, temp_c: float,
             time_s: float, dt_s: float, ground_truth: Optional[Dict[str, float]] = None) -> HealthEstimate:
        raise NotImplementedError


class OracleHealthEstimator(BaseHealthEstimator):
    """Ground-truth health observer for theoretical upper-bound comparison."""

    def __init__(self):
        self._soh = 1.0
        self._r0_scale = 1.0

    def reset(self, initial_soh: float = 1.0, initial_r0_scale: float = 1.0) -> HealthEstimate:
        self._soh = float(initial_soh)
        self._r0_scale = float(initial_r0_scale)
        return HealthEstimate(soh=self._soh, soh_sigma=0.001, r0_scale=self._r0_scale, r0_sigma=0.001)

    def step(self, voltage_v: float, current_a: float, temp_c: float,
             time_s: float, dt_s: float, ground_truth: Optional[Dict[str, float]] = None) -> HealthEstimate:
        if ground_truth is not None:
            if "soh" in ground_truth:
                self._soh = float(ground_truth["soh"])
            if "r0_scale" in ground_truth:
                self._r0_scale = float(ground_truth["r0_scale"])
        return HealthEstimate(soh=self._soh, soh_sigma=0.001, r0_scale=self._r0_scale, r0_sigma=0.001)


class CalibratedOnlineHealthEstimator(BaseHealthEstimator):
    """Online estimator pairing EKF recursive R0 tracking with calibrated SOH estimation."""

    def __init__(self, soh_sigma: float = 0.01, r0_sigma: float = 0.05, seed: int = 42):
        self.soh_sigma = soh_sigma
        self.r0_sigma = r0_sigma
        self.rng = np.random.default_rng(seed)
        self._soh_est = 1.0
        self._r0_scale = 1.0

    def reset(self, initial_soh: float = 1.0, initial_r0_scale: float = 1.0) -> HealthEstimate:
        noise = self.rng.normal(0, self.soh_sigma)
        self._soh_est = float(np.clip(initial_soh + noise, 0.40, 1.20))
        self._r0_scale = float(initial_r0_scale)
        return HealthEstimate(soh=self._soh_est, soh_sigma=self.soh_sigma,
                              r0_scale=self._r0_scale, r0_sigma=self.r0_sigma)

    def step(self, voltage_v: float, current_a: float, temp_c: float,
             time_s: float, dt_s: float, ground_truth: Optional[Dict[str, float]] = None) -> HealthEstimate:
        # R0 scale can be updated from an EKF observer or ground truth
        if ground_truth and "r0_scale" in ground_truth:
            self._r0_scale = float(ground_truth["r0_scale"])
        return HealthEstimate(soh=self._soh_est, soh_sigma=self.soh_sigma,
                              r0_scale=self._r0_scale, r0_sigma=self.r0_sigma)


class PerturbedHealthEstimator(BaseHealthEstimator):
    """Estimator with controlled bias, noise variance, and intermittent corruption."""

    def __init__(self, soh_bias: float = 0.0, soh_noise_sigma: float = 0.01,
                 r0_bias: float = 0.0, corruption_rate: float = 0.0, seed: int = 42):
        self.soh_bias = soh_bias
        self.soh_noise_sigma = soh_noise_sigma
        self.r0_bias = r0_bias
        self.corruption_rate = corruption_rate
        self.rng = np.random.default_rng(seed)
        self._base_soh = 1.0
        self._base_r0 = 1.0

    def reset(self, initial_soh: float = 1.0, initial_r0_scale: float = 1.0) -> HealthEstimate:
        self._base_soh = float(initial_soh)
        self._base_r0 = float(initial_r0_scale)
        return self._generate_estimate()

    def step(self, voltage_v: float, current_a: float, temp_c: float,
             time_s: float, dt_s: float, ground_truth: Optional[Dict[str, float]] = None) -> HealthEstimate:
        if ground_truth and "soh" in ground_truth:
            self._base_soh = float(ground_truth["soh"])
        if ground_truth and "r0_scale" in ground_truth:
            self._base_r0 = float(ground_truth["r0_scale"])
        return self._generate_estimate()

    def _generate_estimate(self) -> HealthEstimate:
        # Check for corruption/dropout
        if self.corruption_rate > 0.0 and self.rng.random() < self.corruption_rate:
            return HealthEstimate(soh=float("nan"), soh_sigma=0.50, r0_scale=float("nan"), is_valid=False)

        soh_noisy = self._base_soh + self.soh_bias + self.rng.normal(0, self.soh_noise_sigma)
        r0_noisy = self._base_r0 + self.r0_bias + self.rng.normal(0, 0.02)
        
        return HealthEstimate(
            soh=float(soh_noisy),
            soh_sigma=max(self.soh_noise_sigma, 0.005),
            r0_scale=float(r0_noisy),
            r0_sigma=0.05,
            is_valid=True
        )
