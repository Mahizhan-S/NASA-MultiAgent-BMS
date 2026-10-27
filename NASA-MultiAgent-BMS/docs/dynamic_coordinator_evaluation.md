# Multi-Agent Coordinator Evaluation Report (Phase H)

## Executive Summary

Phase H investigates whether introducing dynamic or optimization-based coordinator architectures (arbitrating between Charging Speed, Battery Health, and Thermal Regulation) outperforms the simple, deterministic minimum-rule coordinator ($I = \min(I_{\text{speed}}, I_{\text{health}})$). Evaluated across **$N = 90$ episodes per coordinator** over 30 identical unseen validation scenarios and 3 noise seeds ($N = 270$ runs total), the benchmark conclusively demonstrates that **the deterministic Minimum-Rule Coordinator is Pareto-optimal and outperforms more complex dynamic alternatives**.

```
+-------------------------------------------------------------------------------------------------------------------+
|                                        Phase H Coordinator Comparison                                              |
+------------------------------+----------------+----------------+---------------+------------+-----------+---------+
| Coordinator Architecture     | Target Success | Mean Loss (ΔQ) | Mean Time (s) | Peak Temp  | Comp. Cost| Latency |
+------------------------------+----------------+----------------+---------------+------------+-----------+---------+
| Minimum-Rule (Baseline)      | **100.0%**     | **1.821×10⁻⁴** | 2,380.2 s     | **34.63 °C**| **3.4925**| 0.071 ms|
| Dynamic-Priority             | **100.0%**     | 1.825 × 10⁻⁴   | 2,369.2 s     | 34.64 °C   | 3.4968    | 0.070 ms|
| Constrained Convex QP        | **100.0%**     | 1.843 × 10⁻⁴   | **2,365.1 s** | 34.74 °C   | 3.5165    | 0.077 ms|
+------------------------------+----------------+----------------+---------------+------------+-----------+---------+
```

---

## 1. Coordinator Architectures Evaluated

1. **`MinimumRuleCoordinator` (Baseline Architecture)**:
   - Takes the element-wise minimum of proposed currents:
     $$I_{\text{exec}} = \min(I_{\text{speed}}, I_{\text{health}})$$
   - Relies on the Adaptive Health Agent to provide physical degradation derating and deadline-feasibility floor feedback ($I_{\text{min\_req}}$).
2. **`DynamicPriorityCoordinator` (Rule-Based Context Switching)**:
   - Dynamically reallocates arbitration authority based on battery state:
     - **Urgency Precedence**: When $t_{\text{rem}} < 600\,\text{s}$ or feasibility status is `AT_RISK`, overrides health throttling up to $1.15 \times I_{\text{health}}$ to ensure deadline completion.
     - **Thermal Relaxation**: When $T > 38.0^\circ\text{C}$, throttles charging current to $\min(0.80 \cdot I_{\text{speed}}, I_{\text{health}})$.
     - **Nominal Compromise**: Operates under nominal health derating.
3. **`ConstrainedQPCoordinator` (Analytic Convex Optimization)**:
   - Formulates and solves a 1-D convex quadratic program at every control step ($\Delta t = 1.0\,\text{s}$):
     $$\min_{I \ge 0} \quad w_{\text{speed}}(t) (I - I_{\text{speed}})^2 + w_{\text{health}}(SOH, R_0) (I - I_{\text{health}})^2 + w_{\text{thermal}}(T) R_0 I^2$$
     $$\text{s.t.} \quad I_{\text{min\_req}} \le I \le \max(I_{\text{speed}}, I_{\text{health}})$$
   - Weights scale dynamically based on remaining deadline margin, cell resistance growth ($R_0 / R_{0,\text{nom}}$), and temperature stress. Solved analytically in $< 1\,\mu\text{s}$.

---

## 2. Experimental Findings & Statistical Analysis

### 2.1 Head-to-Head Performance & Paired Wilcoxon Tests

| Candidate Coordinator | Compared Against | Metric | Mean Difference $\Delta$ | Cohen's $d$ | Wilcoxon $p$-value | Practical Significance |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Dynamic-Priority** | Minimum-Rule | Composite Cost $J$ | $+0.0043$ | $+0.253$ | $0.0333$ | Statistically worse composite cost. |
| **Dynamic-Priority** | Minimum-Rule | Capacity Loss | $+3.48 \times 10^{-7}$ | $+0.396$ | $0.00007$ | Slightly higher degradation due to urgency current boosts. |
| **Dynamic-Priority** | Minimum-Rule | Charging Time | $-11.0\,\text{s}$ | $-0.376$ | $0.00022$ | Completes 11 seconds faster. |
| **Constrained QP** | Minimum-Rule | Composite Cost $J$ | $+0.0240$ | $+0.300$ | $0.00253$ | Worsens composite cost significantly. |
| **Constrained QP** | Minimum-Rule | Capacity Loss | $+2.18 \times 10^{-6}$ | $+0.446$ | $0.02676$ | **$+1.20\%$ higher degradation** from quadratic current blending. |
| **Constrained QP** | Minimum-Rule | Charging Time | $-15.1\,\text{s}$ | $-0.148$ | $0.21229$ | Marginally faster but not statistically significant. |

---

## 3. Engineering & Scientific Implications

### 3.1 Why the Minimum Rule Wins: Occam's Razor in MA-BMS
1. **Redundancy of Coordinator Compromise**:
   - The Adaptive Health Agent already incorporates the deadline feasibility constraint ($I_{\text{min\_req}}$) and internal resistance ($R_0$) directly into its physical proposal.
   - Therefore, the proposed health ceiling $I_{\text{health}}$ is *already* deadline-feasible whenever physically possible.
   - When a coordinator attempts to "blend" or prioritize speed by taking $I > I_{\text{health}}$, it pushes additional current through an aged, resistive cell, causing unnecessary Joule heating ($I^2 R$) and accelerating capacity loss without improving deadline success (which was already 100%).
2. **Safety and Determinism**:
   - The Minimum-Rule coordinator is unconditionally deterministic, highly interpretable, and computationally negligible ($< 0.071\,\text{ms/step}$).
   - Dynamic prioritization and QP optimization introduce tuning hyperparameters (urgency horizons, weight scaling exponents) that offer no net benefit over well-designed physical agents.
3. **Recommendation for Production Deployment**:
   - Retain the **Minimum-Rule Coordinator** as the production baseline.
   - Reserve dynamic coordination only for multi-pack load balancing or heterogeneous cell fleet scheduling, where inter-agent resource contention actually exists.

---

## 4. Generated Artifacts

- **`results/coordinator_comparison.csv`**: Full summary statistics for all three coordinators across 90 episodes each.
- **`results/coordinator_paired_tests.csv`**: Paired Wilcoxon signed-rank tests, Cohen's $d$, and 95% confidence intervals.
- **`results/coordinator_performance_comparison.png`**: Three-panel boxplot comparing composite cost $J$, capacity degradation ($\times 10^{-4}$), and charging time across coordinator architectures.
