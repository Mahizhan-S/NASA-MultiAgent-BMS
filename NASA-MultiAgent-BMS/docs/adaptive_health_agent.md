# Adaptive Health Agent Specification & Hardening (Phase B)

**Document Version:** 2.0.0  
**Module:** `mabms.agents.health_agent.HealthAgent`  
**Standard:** Physics-Informed Multi-Factor Aging Degradation & Constraint-Satisfaction Control  
**Status:** Validated & Accepted (39/39 Unit & Integration Tests Passing)  

---

## 1. Overview & Problem Definition

In earlier iterations of the multi-agent battery management system, the health proposer enforced a static ceiling on the charging C-rate:
$$C_{\mathrm{cap}} = C_{\mathrm{base\_max}} \cdot \left[0.5 + 0.5 \cdot \mathrm{clip}\left(\frac{\mathrm{SOH} - \mathrm{SOH}_{\mathrm{eol}}}{1 - \mathrm{SOH}_{\mathrm{eol}}},\, 0,\, 1\right)\right]$$

### Flaws in the Static Formulation
1. **Current Starvation of Aged Cells:** As internal resistance ($R_0$) grows due to solid-electrolyte interphase (SEI) growth, charging takes longer. Capping current strictly by SOH starved the cell of necessary charge, causing deadline success to drop to **65.6%** on aged packs.
2. **Disconnection from Physical Degradation Mechanisms:** Degradation accelerates primarily from:
   - Elevated temperature ($I^2 R_0$ Joule heating entering Arrhenius acceleration).
   - High anode overpotentials at $\mathrm{SOC} > 80\%$ leading to lithium plating.
   - High cell terminal voltage ($V_t > 4.15\,\text{V}$) causing cathode transition-metal dissolution.
   A static C-rate limit ignored these dynamic state variables.

---

## 2. Mathematical Formulation (Phase B)

### 2.1. First-Order Deadline Feasibility Floor
To guarantee that the primary user requirement (charging the battery before the operational deadline) is met, we compute the first-order average required current in consistent SI units:

$$I_{\mathrm{required}} = \frac{3600 \cdot Q_{\mathrm{nom}} \cdot \max(\mathrm{SOC}_{\mathrm{target}} - \mathrm{SOC},\, 0.0)}{\eta_{\mathrm{chg}} \cdot \max(t_{\mathrm{deadline}} - t,\, 1.0)} \quad [\text{Amperes}]$$

$$C_{\mathrm{required}} = \frac{I_{\mathrm{required}}}{Q_{\mathrm{nom}}} \quad [\text{C-rate}]$$

$$C_{\mathrm{deadline}} = C_{\mathrm{required}} \cdot \gamma_{\mathrm{margin}} \quad (\gamma_{\mathrm{margin}} = 1.15)$$

where:
- $Q_{\mathrm{nom}} = 2.0\,\text{Ah}$ is the nominal cell capacity.
- $\eta_{\mathrm{chg}} = 0.98$ is the coulombic charging efficiency.
- $t_{\mathrm{deadline}} - t$ is the remaining charging duration in seconds.

### 2.2. Feasibility Classification
The controller classifies the operational envelope into three mutually exclusive operational states:

$$\text{Status} = \begin{cases}
\mathbf{FEASIBLE}, & \text{if } I_{\mathrm{required}} \le I_{\mathrm{health\_max}} \text{ and } I_{\mathrm{required}} \le I_{\mathrm{hard\_limit}} \\
\mathbf{AT\_RISK}, & \text{if } I_{\mathrm{health\_max}} < I_{\mathrm{required}} \le I_{\mathrm{hard\_limit}} \\
\mathbf{INFEASIBLE}, & \text{if } I_{\mathrm{required}} > I_{\mathrm{hard\_limit}} \text{ or } (t_{\mathrm{rem}} \le 0 \text{ with } \Delta\mathrm{SOC} > 0)
\end{cases}$$

- **Feasible:** Gentle charging minimizes degradation while comfortably meeting deadline.
- **At-Risk:** Deadline is tight; the health agent temporarily expands its current ceiling towards $C_{\mathrm{deadline}}$ to guarantee 100% on-time completion without starvation.
- **Infeasible:** The deadline is physically impossible even at absolute hardware current limits ($4.90\,\text{A}$). The controller safely saturates at $I_{\mathrm{hard\_limit}}$ and notifies telemetry without numerical explosion.

---

### 2.3. Constrained Candidate-Current Selection
The health current ceiling $C_{\mathrm{health\_max}}$ is formulated as the maximum allowable current consistent with keeping predicted incremental degradation within a gentle aging budget:

$$C_{\mathrm{health\_max}} = C_{\mathrm{base\_max}} \cdot f_{\mathrm{SOH}} \cdot f_{\rho} \cdot f_T \cdot f_{\mathrm{SOC}}$$

where:
1. **Uncertainty-Aware SOH Margin ($f_{\mathrm{SOH}}$):**
   Using the lower $95\%$ confidence bound on health estimate:
   $$\mathrm{SOH}_{\mathrm{eff}} = \max(\mathrm{SOH} - 1.96 \cdot \sigma_{\mathrm{SOH}},\, 0.01)$$
   $$f_{\mathrm{SOH}} = 0.6 + 0.4 \cdot \mathrm{clip}\left(\frac{\mathrm{SOH}_{\mathrm{eff}} - \mathrm{SOH}_{\mathrm{eol}}}{1.0 - \mathrm{SOH}_{\mathrm{eol}}},\, 0.0,\, 1.0\right)$$
2. **Resistance Overpotential Derating ($f_\rho$):**
   Bounds $I^2 R_0$ Joule heating as internal resistance increases:
   $$f_\rho = \frac{1.0}{\sqrt{\max(\hat{\rho}_{\mathrm{filtered}},\, 1.0)}}$$
3. **Continuous Arrhenius Thermal Derating ($f_T$):**
   $$f_T = \begin{cases} 1.0, & \text{if } T_{\mathrm{filtered}} \le 38.0^\circ\text{C} \\ \exp\left(-0.08 \cdot (T_{\mathrm{filtered}} - 38.0^\circ\text{C})\right), & \text{if } T_{\mathrm{filtered}} > 38.0^\circ\text{C} \end{cases}$$
4. **High-SOC Overpotential Derating ($f_{\mathrm{SOC}}$):**
   Suppresses lithium plating at high anode occupancy:
   $$f_{\mathrm{SOC}} = \begin{cases} 1.0, & \text{if } \mathrm{SOC} \le 0.80 \\ 1.0 - 0.35 \cdot \left(\frac{\mathrm{SOC} - 0.80}{0.20}\right)^2, & \text{if } \mathrm{SOC} > 0.80 \end{cases}$$

---

### 2.4. Noise Smoothing and Slew-Rate Limiting
1. **Exponential Moving Average (EMA) Smoothing:**
   $$\hat{\rho}_{\mathrm{filtered}, k} = (1 - \alpha_\rho) \hat{\rho}_{\mathrm{filtered}, k-1} + \alpha_\rho \hat{\rho}_k \quad (\alpha_\rho = 0.20)$$
   $$T_{\mathrm{filtered}, k} = (1 - \alpha_T) T_{\mathrm{filtered}, k-1} + \alpha_T T_k \quad (\alpha_T = 0.20)$$
2. **Current Slew-Rate Limiter:**
   $$\left|\frac{C_k - C_{k-1}}{\Delta t}\right| \le \dot{C}_{\mathrm{max}} \quad (\dot{C}_{\mathrm{max}} = 0.50\,\text{C/s})$$
   Eliminates high-frequency current step transients, reducing $V_1$ diffusion shocks and chattering.

---

## 3. Online Incremental Degradation Predictor

For every proposal, the agent computes the expected single-step capacity loss:
$$d\mathrm{Loss} = k_{\mathrm{deg}} \cdot \left(\frac{C_{\mathrm{prop}} \cdot Q_{\mathrm{nom}} \cdot \Delta t}{3600 \cdot Q_{\mathrm{nom}}}\right) \cdot f_T(T) \cdot \left(1 + \beta_C \max(C_{\mathrm{prop}} - 0.5,\, 0)\right) \cdot \left(1 + \beta_S \left(\frac{\max(\mathrm{SOC} - 0.8,\, 0)}{0.2}\right)^2\right)$$
Units: Fractional capacity loss per step (dimensionless $\Delta Q / Q_{\mathrm{nom}}$).

---

## 4. Phase B Acceptance Verification Matrix

| Test Category | Test Case | Target Requirement | Measured Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Aging Derating** | Fresh vs. Aged Cell | $C_{\mathrm{aged}} \le C_{\mathrm{fresh}}$ | $0.85\,\text{C}$ vs $1.50\,\text{C}$ | ✅ PASS |
| **Impedance Derating** | Elevated Resistance ($\hat{\rho} = 2.2$) | $C_{\mathrm{high\_\rho}} < C_{\mathrm{norm\_\rho}}$ | $1.01\,\text{C}$ vs $1.50\,\text{C}$ | ✅ PASS |
| **Thermal Protection** | High Temp ($42.5^\circ\text{C}$ vs $25^\circ\text{C}$) | $C_{\mathrm{hot}} < C_{\mathrm{cool}}$ | $1.04\,\text{C}$ vs $1.50\,\text{C}$ | ✅ PASS |
| **Feasibility Class** | Relaxed Deadline ($5000\,\text{s}$) | $\text{Status} = \mathbf{FEASIBLE}$ | `FEASIBLE` | ✅ PASS |
| **Feasibility Class** | Impossible Deadline ($10\,\text{s}$) | $\text{Status} = \mathbf{INFEASIBLE}$ | `INFEASIBLE` | ✅ PASS |
| **Slew-Rate Limit** | Rapid Transient Demand | Jump $\le \dot{C}_{\mathrm{max}} \cdot \Delta t$ | $\Delta C \le 0.50\,\text{C/s}$ | ✅ PASS |
| **Uncertainty Aware** | $\sigma_{\mathrm{SOH}} = 0.08$ vs $0.005$ | $C_{\sigma=0.08} \le C_{\sigma=0.005}$ | Conservative reduction | ✅ PASS |
| **Expired Deadline** | $t_{\mathrm{deadline}} - t \le 0$ | No division by zero, finite | `INFEASIBLE`, finite output | ✅ PASS |
| **Zero Demand** | $\Delta \mathrm{SOC} \le 0$ | Current zeroed, on-target | `FEASIBLE`, $I_{\mathrm{req}} = 0$ | ✅ PASS |
| **Corrupted State** | $\mathrm{SOC} = \mathrm{NaN}$, $V_t = \mathrm{Inf}$ | Deterministic fallback | Fallback active, zero violation | ✅ PASS |
| **Regression Suite** | All 39 Pytest Cases | Zero test regressions | 39 passed in 11.93s | ✅ PASS |
