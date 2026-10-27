"""v7.2: ageing-dependent series resistance. Drop into mabms/environment/.
R0_eff(soh) = R0_new * (1 + k_r0 * (1 - soh_cap)),  soh_cap = capacity retention (1.0 = new).
Calibration (scripts 22): R0_new ~ 0.034 ohm (A_early=4.53C, h=0.068, I=3A); k_r0 ~ 2.6-3.5 (r=0.54 across 26 packs).
Use per-episode random k_r0 ~ U(1.5, 4.5) for robustness; do NOT treat 3.0 as precise.
"""


def r0_scale(soh_cap: float, k_r0: float = 3.0) -> float:
    return 1.0 + k_r0 * max(0.0, 1.0 - soh_cap)
