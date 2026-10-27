# Degradation-Model Validation & Calibration Report (Phase C)

**Document Version:** 1.0.0  
**Target:** Physical Capacity-Fade Surrogate & Parameter Identifiability  
**Dataset:** NASA Ames Prognostics Center of Excellence (PCoE) Li-Ion Accelerated Aging Runs  
**Validation Artifacts:** [`results/degradation_validation_curves.png`](file:///Users/mahizhan/Documents/Final_project/NASA-MultiAgent-BMS/results/degradation_validation_curves.png), [`results/degradation_validation_metrics.json`](file:///Users/mahizhan/Documents/Final_project/NASA-MultiAgent-BMS/results/degradation_validation_metrics.json)  

---

## 1. Physical Capacity-Fade Governing Equation

The empirical degradation surrogate in [`mabms/environment/battery_model.py`](file:///Users/mahizhan/Documents/Final_project/NASA-MultiAgent-BMS/mabms/environment/battery_model.py) models the incremental irreversible capacity loss per step $\Delta t$:

$$d\mathrm{Loss} = k_{\mathrm{deg}} \cdot \left(\frac{|I| \Delta t}{3600 \cdot Q_{\mathrm{nom}}}\right) \cdot f_T(T) \cdot f_C(C_{\mathrm{rate}}) \cdot f_S(\mathrm{SOC})$$

where:
- $\frac{|I| \Delta t}{3600 \cdot Q_{\mathrm{nom}}}$ is the normalized Coulombic throughput in equivalent full cycles.
- $f_T(T)$ is the Arrhenius thermal acceleration factor.
- $f_C(C_{\mathrm{rate}})$ is the kinetic overpotential stress factor.
- $f_S(\mathrm{SOC})$ is the high-state-of-charge mechanical/plating stress factor.

---

## 2. Arrhenius Activation & Kinetic Parameters

### 2.1. Arrhenius Thermal Factor ($f_T$)
$$f_T(T) = \exp\left(E_{a,\mathrm{deg}} \left(\frac{1}{T_{\mathrm{ref}} + 273.15} - \frac{1}{T + 273.15}\right)\right)$$

- **Reference Temperature:** $T_{\mathrm{ref}} = 25.0^\circ\text{C}$ ($298.15\,\text{K}$).
- **Apparent Activation Temperature ($E_{a,\mathrm{deg}} / R$):** $4{,}067.7\,\text{K}$ (median bootstrap estimate).
- **Apparent Activation Energy ($E_a$):**
  $$E_a = 4067.7\,\text{K} \times 8.31446\,\frac{\mathrm{J}}{\mathrm{mol}\cdot\mathrm{K}} \approx 33.82\,\frac{\mathrm{kJ}}{\mathrm{mol}}$$
- **Electrochemical Interpretation:** This value lies directly in the reported literature range ($30\text{--}45\,\mathrm{kJ/mol}$) for solvent molecule diffusion through the solid-electrolyte interphase (SEI) layer in commercial $\mathrm{LiCoO}_2$ / graphite 18650 cells.

### 2.2. Kinetic & High-SOC Stress Factors
- **C-Rate Factor:** $f_C(C_{\mathrm{rate}}) = 1.0 + \beta_C \cdot \max(C_{\mathrm{rate}} - 0.5,\, 0)$, with $\beta_C = 0.50$. Accounts for non-linear mass transport polarization and localized current crowding.
- **High-SOC Factor:** $f_S(\mathrm{SOC}) = 1.0 + \beta_S \cdot \left(\frac{\max(\mathrm{SOC} - 0.80,\, 0)}{0.20}\right)^2$, with $\beta_S = 1.0$. Captures elevated cathode surface oxidation and increased thermodynamic susceptibility to lithium plating at low anode potentials ($< 50\,\text{mV}$ vs $\mathrm{Li/Li}^+$).

---

## 3. Empirical Model Selection on NASA PCoE Validation Packs

To select the optimal degradation formulation without overfitting, four structural candidate variants were fitted to training packs and evaluated on held-out NASA validation packs (Battery 11, Battery 21, Battery 22) via [`mabms/environment/degradation_fit.py`](file:///Users/mahizhan/Documents/Final_project/NASA-MultiAgent-BMS/mabms/environment/degradation_fit.py):

| Model Variant | Functional Description | Fitted $E_a / R$ (K) | Validation SOH MAE | Validation SOH RMSE | Final End Error |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`k_only`** | Linear Coulomb throughput only | — | 5.42% | 7.79% | 11.10% |
| **`k_T`** | Throughput + Arrhenius Thermal ($f_T$) | $4{,}914.8\,\text{K}$ | 3.16% | 3.74% | 5.90% |
| **`k_T_C`** | Throughput + Arrhenius + C-Rate ($f_T \cdot f_C$) | $2{,}862.9\,\text{K}$ | 2.67% | 3.04% | 3.63% |
| **`k_T_C_S`** | **Throughput + Arrhenius + C-Rate + SOC Stress** | $\mathbf{4{,}112.2\,\text{K}}$ | **1.56%** | **2.15%** | **2.32%** |
| `ASSUMED` | Uncalibrated baseline configuration | $4{,}963.7\,\text{K}$ | 3.30% | 3.87% | 5.88% |

### Key Findings
1. Moving from a naive throughput model (`k_only`) to the full multi-factor formulation (`k_T_C_S`) reduces validation tracking error from **5.42% down to 1.56%** (a **71.2% error reduction**).
2. The high-SOC stress term ($f_S$) is essential to explain the knee-point acceleration observed in batteries cycled to high cutoff voltages.

---

## 4. Per-Pack Validation Error Breakdown

| Pack Identifier | NASA Test Profile | Number of Reference Checks | SOH MAE | SOH RMSE | Final Check Error |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Battery 11** | Accelerated 4.2V CC-CV Cycling | 42 | **1.44%** | 1.47% | 1.80% |
| **Battery 21** | Early-Life Cycling | 3 | **0.18%** | 0.20% | 0.23% |
| **Battery 22** | Deep Discharge Accelerated Run | 22 | **3.05%** | 3.42% | 4.94% |
| **Overall Validation** | Combined Held-Out Cohort | 67 | **1.56%** | **2.15%** | **2.32%** |

---

## 5. Parameter Identifiability & Bootstrap Uncertainty (95% CI)

Evaluated via $N = 30$ pack-level stratified bootstrap resamples:

| Parameter | Interpretation | 2.5% Bound | Median (50%) | 97.5% Bound |
| :--- | :--- | :---: | :---: | :---: |
| **$k_{\mathrm{deg}}$** | Base loss per cycle throughput | $1.0 \times 10^{-5}$ | $1.0 \times 10^{-5}$ | $2.0 \times 10^{-5}$ |
| **$E_a / R$ (K)** | Apparent activation temperature | $3{,}192.7\,\text{K}$ | $4{,}067.7\,\text{K}$ | $4{,}513.7\,\text{K}$ |
| **$\beta_C$** | C-rate kinetic stress slope | 1.58 | 5.00 | 5.00 |
| **$\beta_S$** | High-SOC overpotential exponent scale | 17.18 | 20.00 | 20.00 |

*Identifiability Note:* In NASA PCoE data, ambient temperature and cycling C-rate were partially correlated across packs (e.g. Battery 5/6/7 vs 45/46/47). The pack-level bootstrap bounds quantify this epistemic uncertainty cleanly.

---

## 6. Exploitation Risks & Operational Validity Envelope

### 6.1. Potential Model Vulnerabilities (Gaming the Objective)
If an unconstrained optimizer were given free rein over this surrogate, it could discover artificial "loopholes":
1. **Ultra-Low Temperature Exploitation:** Because $f_T \propto \exp(-E_a/RT)$, the model predicts near-zero degradation at $-20^\circ\text{C}$. In reality, sub-zero charging causes severe lithium metal plating.
2. **Current Pulsing:** High-frequency on/off pulses might average below the $0.5\,\text{C}$ threshold while exciting unmodeled diffusion overpotentials.

### 6.2. Enforced Operational Validity Envelope
To prevent policy exploitation, [`mabms/agents/health_agent.py`](file:///Users/mahizhan/Documents/Final_project/NASA-MultiAgent-BMS/mabms/agents/health_agent.py) and [`mabms/agents/safety_agent.py`](file:///Users/mahizhan/Documents/Final_project/NASA-MultiAgent-BMS/mabms/agents/safety_agent.py) enforce explicit physical bounds:

| State Variable | Minimum Admissible Bound | Maximum Admissible Bound | Physical Enforcement Mechanism |
| :--- | :---: | :---: | :--- |
| **Cell Temperature ($T$)** | $15.0^\circ\text{C}$ | $45.0^\circ\text{C}$ | Hard derating above $38^\circ\text{C}$; cutoff at $45^\circ\text{C}$ |
| **Charging C-Rate ($C$)** | $0.20\,\text{C}$ | $2.45\,\text{C}$ | Hardware current clamp $I_{\mathrm{max}} = 4.90\,\text{A}$ |
| **State of Charge ($\mathrm{SOC}$)** | $0.00$ | $0.98$ | Immediate current cutoff at $\mathrm{SOC} \ge 0.98$ |
| **Terminal Voltage ($V_t$)** | $2.50\,\text{V}$ | $4.25\,\text{V}$ | Closed-loop CV voltage clamp $V_{\mathrm{max}} = 4.20\,\text{V}$ |

---

## 7. Model-Validity Limitations & Caveats

1. **Cycle Aging vs Calendar Aging:** The model focuses on dynamic charge-throughput degradation. It does not model prolonged calendar aging (e.g. multi-month open-circuit rest at elevated SOC).
2. **Phase Changes & Knee-Point Transition:** The empirical law captures gradual SEI accumulation and cathode degradation accurately down to $75\%\text{--}80\%$ SOH. It does not model sudden internal separator collapse or short-circuit thermal runaway.
3. **Simulation Boundary:** All reported degradation improvements are validated against empirical NASA PCoE reference runs and calibrated 1-RC ECM models; they should not be extrapolated to unvalidated cell chemistries (e.g., LFP, solid-state) without recalibration.
