# Objective Formulation & Trade-Off Sensitivity Analysis (Phase E)

## Executive Summary

Phase E systematically deconstructs the multi-objective cost function $J$ used to evaluate and tune the MA-BMS framework. Through a **10,000-sample Dirichlet Monte Carlo sensitivity analysis** across multiple preference regimes, coupled with **pairwise Pareto dominance mapping** and **cross-objective correlation analysis**, this study resolves the apparent paradox in Chapter 6 where the single-agent policy appeared to dominate composite cost $J$ despite exhibiting higher battery degradation and higher peak temperatures.

```
+------------------------------------------------------------------------------------------------------------------+
|                                  Dirichlet Sensitivity Across Preference Regimes                                  |
+------------------------------+--------------------+------------------------+--------------------+----------------+
| Controller                   | Speed vs Deg (2-D) | Speed+Deg+Thermal (3-D)| Comp. Cost J (3-D) | All 4 Obj (4-D)|
+------------------------------+--------------------+------------------------+--------------------+----------------+
| Conventional CC-CV           |             57.76% |                 43.28% |              1.27% |          1.24% |
| Independent-Deadline         |              2.21% |                  3.11% |              0.23% |          0.67% |
| Static Health-Cap (1.75 A)   |             31.07% |                 46.68% |              3.37% |          7.21% |
| Adaptive Health (Full)       |              8.96% |                  6.93% |              0.39% |          0.38% |
| Single-Agent Policy          |           **0.00%**|               **0.00%**|             94.74% |         90.50% |
+------------------------------+--------------------+------------------------+--------------------+----------------+
```

---

## 1. Mathematical Formulation & Normalization Mechanics

### 1.1 Composite Scalar Objective $J$
The composite objective is defined as:
$$J = w_{\text{time}} \cdot T_{\text{norm}} + w_{\text{deg}} \cdot D_{\text{norm}} + w_{\text{cool}} \cdot E_{\text{norm}} + w_{\text{safe}} \cdot V_{\text{norm}}$$

where the component terms are normalized as:
1. **Charging Duration ($T_{\text{norm}}$)**:
   $$T_{\text{norm}} = \frac{t_{\text{target}}}{t_{\text{deadline}}} \quad (\text{penalized to } 1.0 + \text{fail\_penalty} = 2.0 \text{ if deadline is violated})$$
2. **Cycle Degradation ($D_{\text{norm}}$)**:
   $$D_{\text{norm}} = \frac{\Delta Q_{\text{loss}}}{d_{\text{scale}}}, \quad d_{\text{scale}} = \text{median}(\Delta Q_{\text{conventional}}) \approx 1.565 \times 10^{-4}$$
3. **Auxiliary Cooling Energy ($E_{\text{norm}}$)**:
   $$E_{\text{norm}} = \frac{E_{\text{cooling}}}{e_{\text{scale}}}, \quad e_{\text{scale}} = \text{mean}(E_{\text{cooling, conventional}}) \approx 0.01021\,\text{Wh}$$
4. **Safety Violations ($V_{\text{norm}}$)**:
   $$V_{\text{norm}} = \frac{N_{\text{unsafe}}}{v_{\text{scale}}}, \quad v_{\text{scale}} = 10.0$$
   *(Since all controllers operate with the deterministic safety supervisor active, $V_{\text{norm}} = 0.0$ across all valid runs).*

---

## 2. Objective Deconstruction & Source of the Single-Agent "Win Rate"

### 2.1 The Cooling Energy Normalization Artifact
Inspection of the normalized mean objective vectors reveals the exact mechanism:

| Controller | Mean Time (s) | $T_{\text{norm}}$ | Mean Loss ($\Delta Q$) | $D_{\text{norm}}$ | Cooling Energy (Wh) | $E_{\text{norm}}$ | Peak Temp ($^\circ\text{C}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conventional CC-CV** | **1,768.0 s** | **0.482** | $1.992 \times 10^{-4}$ | 1.273 | $0.0574\,\text{Wh}$ | 5.624 | $35.19^\circ\text{C}$ |
| **Independent-Deadline** | 2,100.6 s | 0.572 | $1.891 \times 10^{-4}$ | 1.208 | $0.0357\,\text{Wh}$ | 3.499 | $34.86^\circ\text{C}$ |
| **Static Health-Cap** | 3,838.5 s | 0.976 | **$1.589 \times 10^{-4}$** | **1.015** | $0.0328\,\text{Wh}$ | 3.217 | **$33.57^\circ\text{C}$** |
| **Adaptive Health (Full)** | 2,406.3 s | 0.648 | $1.814 \times 10^{-4}$ | 1.159 | $0.0343\,\text{Wh}$ | 3.360 | $34.61^\circ\text{C}$ |
| **Single-Agent Policy** | 2,113.7 s | 0.570 | $1.999 \times 10^{-4}$ | 1.277 | **$0.0065\,\text{Wh}$** | **0.642** | $35.64^\circ\text{C}$ |

#### Critical Finding:
1. In absolute physical terms, **$0.01021\,\text{Wh} \approx 36.8\,\text{Joules}$**. In an automotive 400V/800V EV battery pack (delivering $60\text{--}100\,\text{kWh}$ per charge), an auxiliary cooling difference of $0.028\,\text{Wh}$ per cell is negligible compared to the total energy throughput ($> 8.5\,\text{Wh}$ per cell).
2. However, because $e_{\text{scale}}$ is tiny ($0.01021\,\text{Wh}$), the normalized energy term $E_{\text{norm}}$ swings by nearly **$5.0$ units** between controllers ($0.642$ for single-agent vs $3.360$ for adaptive health vs $5.624$ for CC-CV).
3. The Single-Agent Policy was numerically tuned via random search directly against the scalar objective $J$. It discovered a local minimum that **turns off cooling almost entirely**, accepting higher cell temperature ($35.64^\circ\text{C}$) and higher degradation ($1.999 \times 10^{-4}$) to minimize $E_{\text{norm}}$.
4. In contrast, the Multi-Agent Thermal Agent actively regulates cell temperature to the setpoint ($35.0^\circ\text{C}$) to protect long-term cell health, consuming $0.034\,\text{Wh}$. In the scalar objective $J$, this sensible thermal protection is heavily penalized.

---

## 3. Preference Regimes & Core Battery Performance

When the auxiliary cooling energy term is removed or appropriately weighted, the true capability of each architecture becomes clear:

### 3.1 Regime 1: Speed vs. Battery Degradation (2-D Simplex)
Sampling weights $(w_{\text{speed}}, w_{\text{deg}}) \sim \text{Dir}(1, 1)$:
- **Conventional CC-CV wins 57.76%** of the space: Dominates when user preference demands the fastest possible charge ($< 1,800\,\text{s}$) regardless of cell wear.
- **Static Health-Cap wins 31.07%**: Dominates when degradation minimization is the sole priority, but at the cost of prolonged charging times ($3,839\,\text{s}$) and deadline violations.
- **Adaptive Health (Full) wins 8.96%**: Captures the optimal balanced operating regime, providing the lowest degradation among all deadline-feasible controllers without missing charging windows.
- **Single-Agent Policy wins 0.00%**: It is Pareto-dominated on the core battery trade-off plane; it is slower than CC-CV and degrades faster than Adaptive Health.

### 3.2 Regime 2: Speed vs. Degradation vs. Thermal Stress (3-D Simplex)
Sampling weights $(w_{\text{speed}}, w_{\text{deg}}, w_{\text{thermal}}) \sim \text{Dir}(1, 1, 1)$ without cooling energy:
- **Static Health-Cap**: **46.68%** win share.
- **Conventional CC-CV**: **43.28%** win share.
- **Adaptive Health (Full)**: **6.93%** win share.
- **Independent-Deadline**: **3.11%** win share.
- **Single-Agent Policy**: **0.00%** win share.

### 3.3 Regime 3: All Objectives Including Cooling ($J$)
- **Single-Agent Policy**: **94.74%** win share (due to the cooling normalization artifact).
- **Static Health-Cap**: **3.37%**.
- **Conventional CC-CV**: **1.27%**.
- **Adaptive Health (Full)**: **0.39%**.

---

## 4. Pareto Dominance Analysis

Across the full multi-objective vectors $\mathbf{J} = [T_{\text{norm}}, D_{\text{norm}}, E_{\text{norm}}, \text{Temp}_{\text{norm}}]$:
- **Pairwise Dominance Matrix**: All pairwise comparisons evaluate to `False`.
- **Significance**: **Every controller belongs to the mutually non-dominated Pareto frontier**. No single controller completely dominates any other across all criteria:
  - CC-CV defines the **Speed Extreme** ($1,768\,\text{s}$).
  - Static Health-Cap defines the **Degradation/Thermal Extreme** ($1.589 \times 10^{-4}, 33.57^\circ\text{C}$), but is deadline-infeasible.
  - Adaptive Health defines the **Balanced Feasible Frontier** ($1.814 \times 10^{-4}, 34.61^\circ\text{C}, 100\%\text{ deadline success}$).
  - Single-Agent defines the **Auxiliary Energy Extreme** ($0.0065\,\text{Wh}$).

---

## 5. Cross-Objective Correlation Structure

Spearman rank correlation ($\rho$) across all $N = 720$ simulated episodes:

| Metric | Charging Time ($t$) | Capacity Loss ($\Delta Q$) | Cooling Energy ($E_{\text{cool}}$) | Peak Temp ($T_{\text{peak}}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Charging Time ($t$)** | $1.000$ | $-0.068$ | $+0.025$ | $-0.057$ |
| **Capacity Loss ($\Delta Q$)** | $-0.068$ | $1.000$ | $+0.794$ | **$+0.304$** |
| **Cooling Energy ($E_{\text{cool}}$)** | $+0.025$ | **$+0.794$** | $1.000$ | **$+0.794$** |
| **Peak Temp ($T_{\text{peak}}$)** | $-0.057$ | **$+0.304$** | **$+0.794$** | $1.000$ |

### Physical Insights:
1. **Strong Coupling Between Thermal Rise & Cooling Energy ($\rho = +0.794$)**: Higher temperatures trigger aggressive closed-loop cooling, directly confirming proper actuator response in the multi-agent thermal control loop.
2. **Thermal Stress Drives Degradation ($\rho = +0.304$)**: Episodes with elevated peak temperatures directly correlate with increased capacity degradation, validating the Arrhenius degradation mechanics.
3. **Decoupling of Time and Degradation ($\rho = -0.068$)**: Charging duration alone is not a monotonic proxy for battery degradation; aggressive charging causes high degradation in short times, but prolonged current starvation also fails deadlines without conferring additional degradation benefits.

---

## 6. Generated Visual Artifacts

1. **`results/dirichlet_sensitivity.png`**:
   - Left Panel: Grouped bar chart comparing win shares across 3-objective and 4-objective Dirichlet preference spaces.
   - Right Panel: Ternary simplex scatter plot projecting winning controllers across $10,000$ random weight coordinates $(w_{\text{time}}, w_{\text{energy}}, w_{\text{deg}})$.
2. **`results/pareto_frontier.png`**:
   - Subplot 1: Speed vs. Degradation Pareto plane ($t_{\text{target}}$ vs $\Delta Q_{\text{loss}}$).
   - Subplot 2: Thermal Rise vs. Degradation Pareto plane ($T_{\text{peak}}$ vs $\Delta Q_{\text{loss}}$).
   - Shows individual episode scatter clouds alongside star markers representing controller centroids.
