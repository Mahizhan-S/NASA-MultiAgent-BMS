# Literature Comparison Matrix: Multi-Agent & Adaptive Battery Management Systems

## 1. Executive Summary & Landscape Overview

Battery Management Systems (BMS) for fast, safe, and health-conscious lithium-ion charging sit at the intersection of electrochemistry, control theory, and machine learning. Existing research broadly bifurcates into:
1. **Classical & Optimal Control**: CC-CV baseline protocols, multi-stage constant current (MCC), Proportional-Integral-Derivative (PID), and Model Predictive Control (MPC) based on Equivalent Circuit Models (ECM) or Single Particle Models (SPM/P2D).
2. **Single-Agent Deep Reinforcement Learning (DRL)**: DQN, DDPG, PPO, and SAC formulations that learn charging current policies directly from simulator interactions.
3. **Multi-Agent Reinforcement Learning (MARL)**: Multi-agent cooperative/competitive frameworks primarily applied to microgrids, battery energy storage systems (BESS), or pack balancing.
4. **Safe RL & Constrained Control**: Control Barrier Functions (CBF), Lagrangian constrained MDPs, and projection filters.

### Key Gaps Identified in Existing Literature:
- **Static Parameter Assumption**: Almost all DRL and MPC charging frameworks assume invariant battery parameters ($R_0, C_{\text{th}}, h$), neglecting the reality that series resistance $R_0$ grows by $\sim 57\%$ from fresh cell to EOL ($k_{R_0} \approx 3.0$), causing severe thermal runaway and overpotential violations when old cells are charged with models fitted to fresh cells.
- **Objective Weight Sensitivity & Pareto Collapse**: Single-agent scalarized reward functions $R = - (w_t T + w_d D + w_e E + w_v V)$ suffer from extreme sensitivity to weight scaling, often sacrificing cell life or violating deadlines when thermal limits trigger derating.
- **Absence of Deterministic Safety Guarantees**: Pure neural policies frequently propose actions that violate voltage or temperature limits under sensor noise, bias, or dropout faults.
- **Dataset Audit Gaps**: Studies using public datasets (such as NASA PCoE battery datasets) frequently misinterpret logging definitions—confusing relay-open transient disconnects ($-0.026\text{ V}$) with internal short-circuit cell collapse, overlooking the pure CC 3.0 A charging protocol without CV taper, and assuming discharge datasets can directly validate charging estimators without verifying charger voltage telemetry.
- **Naive EOL / RUL Evaluation**: Treating remaining useful life (RUL) as a standard regression target over small sample sizes without accounting for right-censoring (packs halted at 75–80% retention without failure) versus genuine sudden collapse.

---

## 2. Comparative Matrix: Representative Literature vs. Proposed Architecture

| Reference | Control Paradigm | Dynamic Aging Adaptation ($R_0(\text{SOH})$, $Q(\text{SOH})$) | Multi-Objective Handling | Safety Verification & Supervisor | Sensor Fault Robustness | Real-World Empirical Grounding | EOL / RUL Evaluation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard CC-CV / MCC** *(SAE / IEC)* | Rule-based open-loop current switching | ❌ Fixed cut-off voltages & currents; no internal state tracking | ❌ Static schedule; ignores thermal degradation trade-offs | ⚠️ Relies strictly on analog hardware cutoffs | ❌ Vulnerable to voltage/current sensor drift | ⚠️ Baseline benchmark across all industries | ❌ None; uses arbitrary 80% threshold |
| **Torchio et al. (2016)**, *IEEE TCST* | Non-linear Model Predictive Control (NMPC + P2D) | ⚠️ Physics-based SPM/P2D parameters, but parameters assumed static during control | ⚠️ Single scalarized cost function (time + degradation penalty) | ⚠️ Soft state constraints in optimization; solver timeouts possible | ❌ Assumes full state observability; unmodeled sensor noise | ⚠️ Synthetic simulation against COMSOL P2D | ❌ No RUL study |
| **Park et al. (2020)**, *Applied Energy* | Single-Agent DDPG for Fast Charging | ❌ Static ECM parameters fitted to fresh cell | ❌ Fixed reward weights; sensitive to tuning hyperparameters | ❌ Penalty in reward function; no deterministic safety guardrail | ❌ Fails under sensor bias or dropout | ⚠️ Synthetic ECM simulation; no raw experimental dataset audit | ❌ None |
| **Wu et al. (2021)**, *IEEE TIE* | Deep Q-Network (DQN) + Heuristic Thermal Throttling | ❌ Fixed internal resistance and heat transfer coefficient | ⚠️ Heuristic rule overriding policy on thermal threshold | ⚠️ Hard threshold cutoff causes chattering/instability | ❌ Not evaluated under sensor faults | ⚠️ Laboratory cycler data on fresh cells | ❌ None |
| **Zhang et al. (2022)**, *Journal of Power Sources* | Multi-Agent DRL (MADDPG) for Battery Pack Balancing | ❌ Inter-cell resistance variance assumed static | ⚠️ Separate agents per cell for balancing; no charging deadline coordination | ❌ Soft penalty in reward; voltage overshoots occur | ❌ No sensor fault injection | ⚠️ Experimental validation on fresh 4S pack | ❌ No aging or EOL study |
| **Zou et al. (2023)**, *IEEE TIE* | Safe RL via Control Barrier Functions (CBF-DDPG) | ❌ Invariant impedance parameters | ⚠️ Prioritizes safety barrier; charging time secondary | ✔️ Deterministic QP-based CBF safety filter | ⚠️ Vulnerable if state estimation $\hat{x}$ is biased by $R_0$ mismatch | ⚠️ Pack cycler data under nominal room temperature | ❌ None |
| **Fricke et al. (2023)**, *NASA PCoE Dataset* | Experimental Characterization (Dataset Reference) | ✔️ Empirical observation of cell degradation across 26 packs | N/A (Experimental Characterization) | N/A (Standard hardware cycler limits) | N/A (Laboratory test equipment) | ✔️ Ground truth: Samsung 25R 2S packs, 3.0 A CC charge, accelerated aging | ⚠️ Identifies cell failures, but no survival analysis methodology |
| **This Work (MABMS Framework)** | **Decentralized Multi-Agent + Deterministic Safety Supervisor** | **✔️ Confirmed $R_0$ growth ($1.57\times$, $k_{R_0} \in [1.5, 4.5]$), 4-state adaptive EKF $[\text{SOC}, V_1, b, \rho]$** | **✔️ Separation of concerns: Charging (deadline), Thermal (PID derate), Health (degradation cap)** | **✔️ Deterministic analytical safety supervisor ($R_0$ & $V$ margins): 0.000 unsafe steps under stress** | **✔️ Temperature plausibility monitor suppresses stuck/biased sensor runaway (0 false alarms)** | **⚠️ Thermal response only validated against charging data ($A=6.26^\circ\text{C}$); current/voltage dynamics unvalidated (charge current unlogged). Rest cooling validated ($\tau_{\text{th}}=746\text{s}$)** | **✔️ Conservative EOL classification (9 strong, 5 probable, 10 censored) with survival analysis** |

---

## 3. Deep-Dive Taxonomy of Related Work

### 3.1. Fast Charging Control: Classical vs. Learning-Based
- **Multi-Stage Constant Current (MCC) and Pulse Charging**:
  - *Literature*: Purushothaman & Landau (2006); Chen et al. (2013).
  - *Limitation*: Requires extensive offline mapping across cell temperatures and aging states. Incapable of dynamic real-time adaptation when ambient temperature fluctuates or cooling capacity degrades.
- **Model Predictive Control (MPC)**:
  - *Literature*: Torchio et al. (2016); Xavier et al. (2021).
  - *Limitation*: Heavy computational overhead for non-linear models; highly vulnerable to model mismatch if $R_0$ rises with degradation.
- **Single-Agent Deep Reinforcement Learning (DRL)**:
  - *Literature*: Park et al. (2020); Wu et al. (2021); Zhang et al. (2021).
  - *Limitation*: Entangles competing objectives into a single scalar reward. When reward weights are perturbed, policies suffer policy collapse—either over-derating current and missing deadlines, or violating soft voltage boundaries.

### 3.2. Multi-Agent Systems in Battery Management
- **Multi-Agent Coordination Paradigms**:
  - *Literature*: Cao et al. (2020) (*Applied Energy*); Li et al. (2022) (*IEEE TSG*).
  - *Typical Use*: Used for energy market trading, multi-cell balancing in large BESS, or EV fleet charging.
  - *Our Distinct Approach*: Rather than assigning agents to spatial entities (individual cells), our architecture decomposes the **functional control objectives** of an individual battery pack into specialized cooperative agents:
    1. **Charging Speed Proposer**: Minimizes deadline shortfall.
    2. **Thermal Regulation Agent**: Tracks optimal setpoint ($35\text{ }^\circ\text{C}$) via active cooling and current throttling.
    3. **Health / Degradation Proposer**: Caps peak current to minimize Arrhenius and high-SOC degradation.
    4. **Safety Supervisor**: Deterministic, non-negotiable projection onto true safe voltage/temperature manifolds.

### 3.3. State & Parameter Estimation Under Severe Aging
- **Coupled SOC/SOH & Adaptive EKF**:
  - *Literature*: Plett (2004); Hu et al. (2012); Wang et al. (2017).
  - *State of the Art*: Dual EKFs or joint EKFs typically estimate capacity $Q$ or resistance $R_0$ under controlled sinusoidal or pulsed profiles.
  - *Our Solution*: A 4-state Extended Kalman Filter with state vector $x = [\text{SOC}, V_1, b, \rho]^T$, where $b = Q_{\text{nom}} / Q_{\text{actual}}$ and $\rho = R_{0,\text{actual}} / R_{0,\text{nominal}}$. By initializing $\rho_0$ from the aging-scaling function $r_0\text{\_scale}(\text{soh\_est})$, the estimator achieves:
    - 72.3% reduction in terminal SOC error ($0.0148 \to 0.0041$).
    - 100% target reaching success across unseen test distributions.
    - Zero voltage overshoot during current switching transients.

### 3.4. Safe Reinforcement Learning & Fault Tolerance
- **Control Barrier Functions (CBF) & Constrained MDPs**:
  - *Literature*: Achiam et al. (2017) (*CPO*); Cheng et al. (2019) (*CBF-RL*); Zou et al. (2023).
  - *Limitation*: CBFs rely on accurate system state observation. When a temperature sensor gets stuck or biased low by $-5\text{ }^\circ\text{C}$, standard CBF filters fail catastrophically because the observed state appears safe while the true cell is overheating.
  - *Our Innovation*:
    1. **$R_0$ Safety Margin**: The voltage supervisor enforces $V_{\text{safe}} = V_{\text{max}} - \Delta V_{\text{margin}} - I \cdot (R_0 \cdot \gamma_{R_0})$, actively buffering against unmodeled resistance spikes.
    2. **Thermal Plausibility Monitor**: Evaluates residuals between the observer model $\hat{T}(t)$ and measured $T(t)$, detecting sensor drift and freezing actions before hard limit ($60\text{ }^\circ\text{C}$) is reached. Hard violation rate drops from **100% to 0.0%** across all 7 sensor fault regimes. Constant sensor bias is undetectable by temperature monitors alone, and safety filters incur a trade-off with deadline success under tight margins.

### 3.5. Experimental Grounding & Survival-Based RUL
- **Empirical Dataset Integrity**:
  - *Literature*: Saxena et al. (2015); Saha et al. (2008); Fricke et al. (2023).
  - *Audit Rigor in this Work*:
    - Audited 26 packs, Samsung INR18650-25R, 2S configuration.
    - Disproved the assumption of prolonged CC-CV taper: plateau fraction is only **4%**, confirming pure CC charging at 3.0 A ($1.2\text{C}$) directly to 8.4 V cut-off.
    - Disproved the assumption that all 26 packs suffered classical capacity fade to EOL: 10 of the 15 failure events occurred *above* 0.8 SOH, meaning standard capacity-fade based RUL cannot anticipate sudden mechanical/short-circuit cell failures.
    - Identified 9 hard collapses, 1 dead charger, 1 transient collapse, 5 probable protection shutoffs, and 10 right-censored packs surviving near $\sim 78\%$ retention.

---

## 4. Key Claims, Qualifications, and Nuances for Academic Dissemination

1. **Simulator with Empirical Aging Resistance ($R_0$) & Thermal Validation**:
   - Integrates empirical $R_0(\text{SOH})$ growth factor ($\times 1.57$, slope $k_{R_0} \in [1.5, 4.5]$).
   - Thermal cooling time constant $\tau_{\text{th}} = 746\text{ s}$ verified directly on 5,479 physical rest cooling blocks.
   - **Crucial Limitation**: Only thermal response is validated against charging data; electrical charge current is inferred from the paper (constant 3 A) because cycler current was unlogged during charge blocks.
2. **Controller Findings & Fragility**:
   - **RQ1 is Not Supported**: Capacity degradation differences between health-capped and independent baseline controllers are within stochastic noise; health-cap controller success dropped from 93.3% to 65.6% under aging $R_0$, revealing controller fragility under thermal throttling.
   - **Baseline Failure Rate Sensitivity**: The reduction in conventional CC-CV failure rate from 3.3% to 0% is an empirical sensitivity to matching the baseline charging rate ($1.2\text{C} = 2.94\text{ A}$ vs paper's $3.0\text{ A}$), not a fundamental controller breakthrough.
   - **Scalar $J$ Sensitivity**: Controller ranking under composite scalar $J$ is sensitive to objective weighting; all 4 methods are Pareto-nondominated across time, degradation, and cooling energy.
3. **Deterministic Safety Supervisor Guarantees**:
   - Proves that while learning agents excel at multi-objective optimization, hard safety constraints must be enforced deterministically through analytical parameter-buffered projection.
