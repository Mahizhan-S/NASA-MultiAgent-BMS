"""SOC estimation: 1-RC equivalent-circuit EKF (state [soc, v1]).
Convention: current > 0 is charge. Terminal voltage V = OCV(soc) + I*R0 + V1.
NOTE: the NASA data have NO SOC ground truth; this estimator is validated only against simulator truth
or against coulomb-counted reference segments, never against 'measured SOC'."""
from __future__ import annotations
from typing import Sequence
import numpy as np


class OCVCurve:
    def __init__(self, soc: Sequence[float], v: Sequence[float]):
        self.soc = np.asarray(soc, float); self.v = np.asarray(v, float)

    def ocv(self, soc: float) -> float:
        return float(np.interp(soc, self.soc, self.v))

    def docv(self, soc: float, eps: float = 1e-3) -> float:
        return (self.ocv(min(soc + eps, 1.0)) - self.ocv(max(soc - eps, 0.0))) / (min(soc + eps, 1.0) - max(soc - eps, 0.0))

    def inverse(self, v: float) -> float:
        return float(np.interp(v, self.v, self.soc))


class SOCEstimator:
    def __init__(self, ocv: OCVCurve, q_ah: float, r0: float, r1: float, tau_s: float,
                 soc0: float, p0=(0.05 ** 2, 0.01 ** 2), q_proc=(1e-9, 1e-7), r_meas=0.01 ** 2,
                 eta: float = 1.0):
        self.ocv, self.q_ah, self.r0, self.r1, self.tau = ocv, q_ah, r0, r1, tau_s
        self.x = np.array([soc0, 0.0]); self.P = np.diag(p0).astype(float)
        self.Q = np.diag(q_proc).astype(float); self.R = float(r_meas); self.eta = eta

    def set_capacity(self, q_ah: float) -> None:
        self.q_ah = q_ah

    @property
    def soc(self) -> float:
        return float(self.x[0])

    @property
    def v1(self) -> float:
        return float(self.x[1])

    def step(self, i_a: float, v_meas: float, dt_s: float) -> float:
        a = np.exp(-dt_s / self.tau)
        # predict
        soc_p = self.x[0] + self.eta * i_a * dt_s / (3600.0 * self.q_ah)
        v1_p = a * self.x[1] + self.r1 * (1 - a) * i_a
        F = np.diag([1.0, a])
        P = F @ self.P @ F.T + self.Q
        # update
        x_p = np.array([soc_p, v1_p])
        H = np.array([[self.ocv.docv(float(np.clip(soc_p, 0, 1))), 1.0]])
        v_pred = self.ocv.ocv(float(np.clip(soc_p, 0, 1))) + i_a * self.r0 + v1_p
        S = float((H @ P @ H.T).item() + self.R)
        K = (P @ H.T / S).ravel()
        self.x = x_p + K * (v_meas - v_pred)
        self.x[0] = float(np.clip(self.x[0], -0.05, 1.05))
        self.P = (np.eye(2) - np.outer(K, H)) @ P
        return self.soc


class AdaptiveSOCEstimator(SOCEstimator):
    """EKF with a third state b = Q_nominal / Q_actual (capacity ratio), random-walk model. Removes the SOC bias that a fixed capacity
    causes on cells whose capacity differs from the nominal one. b is only observable while current flows (not during rest)."""

    def __init__(self, *args, p0_b: float = 0.15 ** 2, q_b: float = 1e-8, b_bounds=(0.6, 1.8), **kw):
        super().__init__(*args, **kw)
        self.x = np.array([self.x[0], 0.0, 1.0])
        self.P = np.diag([self.P[0, 0], self.P[1, 1], p0_b])
        self.Q = np.diag([self.Q[0, 0], self.Q[1, 1], q_b])
        self.b_bounds = b_bounds

    @property
    def b(self) -> float:
        return float(self.x[2])

    def step(self, i_a: float, v_meas: float, dt_s: float) -> float:
        a = np.exp(-dt_s / self.tau)
        k_i = self.eta * i_a * dt_s / (3600.0 * self.q_ah)
        x_p = np.array([self.x[0] + k_i * self.x[2], a * self.x[1] + self.r1 * (1 - a) * i_a, self.x[2]])
        F = np.array([[1.0, 0.0, k_i], [0.0, a, 0.0], [0.0, 0.0, 1.0]])
        P = F @ self.P @ F.T + self.Q
        soc_c = float(np.clip(x_p[0], 0, 1))
        H = np.array([[self.ocv.docv(soc_c), 1.0, 0.0]])
        v_pred = self.ocv.ocv(soc_c) + i_a * self.r0 + x_p[1]
        S = float((H @ P @ H.T).item() + self.R)
        K = (P @ H.T / S).ravel()
        self.x = x_p + K * (v_meas - v_pred)
        self.x[0] = float(np.clip(self.x[0], -0.05, 1.05)); self.x[2] = float(np.clip(self.x[2], *self.b_bounds))
        self.P = (np.eye(3) - np.outer(K, H)) @ P
        return self.soc
