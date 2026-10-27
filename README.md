# Multi-Agent Battery Management System for Predictive Health Estimation, Thermal Regulation, and Safe Charging Control

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-37%2F37%20Passed-brightgreen.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20PCoE%20ALT-orange.svg)](https://data.nasa.gov/)
[![Thesis](https://img.shields.io/badge/Thesis-Amrita%20Vishwa%20Vidyapeetham-red.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete%20%28Single--Agent%20Scope%29-blueviolet.svg)]()

> **Undergraduate / Master's Thesis Project**  
> **Department of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore Campus, India**  
> **Full Compiled Thesis Document**: [`Thesis_Template___Amrita_AID/mythesis.pdf`](Thesis_Template___Amrita_AID/mythesis.pdf) (76 Pages, 14 Figures, 0 Errors)

---

## ⚡ Executive Summary

Lithium-ion battery packs operating under high-stress accelerated mission regimes face competing operational demands: minimizing charging duration, preventing thermal excursion, mitigating capacity fade, and enforcing strict electrochemical safety boundaries.

This repository presents a modular **Multi-Agent Battery Management System (MA-BMS)** evaluated against the **NASA Prognostics Center of Excellence (PCoE) Accelerated Life Testing (ALT)** dataset comprising 26 two-cell in-series (2S) Samsung INR18650-25R NCA packs (**4,918,193 raw telemetry rows** at 1 Hz).

### Key Architectural Pillars:
1. **Decoupled Proposer Agents**: Autonomous agents proposing charging currents optimized for charging speed, thermal regulation, and cycle health.
2. **4-State Adaptive Extended Kalman Filter (EKF)**: Real-time tracking of $[\mathrm{SOC}, V_1, b, \rho]^\top$, where $\rho$ continuously estimates internal series resistance growth ($R_0$) as the battery ages.
3. **Closed-Form Deterministic Safety Supervisor**: An analytical algebraic projection filter enforcing instantaneous voltage and thermal safety boundaries:
   $$I_{\mathrm{safe}} = \frac{V_{\max} - \hat{V}_{\mathrm{OCV}}(\hat{z}) - \Delta V_{\mathrm{margin}}}{\hat{\rho} R_0 + R_{\mathrm{cable}}}$$
   guaranteeing provable zero hard overvoltage violations even under severe sensor noise degradation ($\times 4$) and sensor dropout.
4. **Dual-Track Prognostic Lifecycle Framework**: Non-parametric Kaplan--Meier survival analysis across three operational endpoints paired with Leave-One-Pack-Out (LOPO) Remaining Useful Life (RUL) regression.

---

## 🔬 Key Scientific & Empirical Findings

- **10 of 15 Failure Events Occur Above 0.80 SOH**: Auditing the NASA ALT telemetry revealed that 66.7% of operational breakdowns (sudden internal shorts, voltage collapses) occur while capacity remains strictly above the standard 80% SOH end-of-life threshold, disproving the traditional assumption that gradual capacity fade precedes all catastrophic failures.
- **20.1% Faster Charging**: The tuned single-agent reinforcement learning controller reaches target SOC in **$1{,}435\,\mathrm{s}$ vs $1{,}796\,\mathrm{s}$** for conventional CC-CV, maintaining peak temperature at $38.6^\circ\mathrm{C}$ with **zero unsafe executed steps**.
- **84.8% Pareto Dominance**: Across 10,000 Monte Carlo objective weightings sampled from a symmetric Dirichlet distribution $(w_{\mathrm{time}}, w_{\mathrm{deg}}, w_{\mathrm{energy}}) \sim \mathrm{Dir}(1, 1, 1)$, the single-agent RL controller wins **84.78% of the objective space**.
- **Prognostic Uncertainty Shrinkage**: Ridge regression under Leave-One-Pack-Out cross-validation achieves $\mathrm{MAE} = 113.0\,\mathrm{Ah}$ ($R^2 = 0.732$), with prediction uncertainty dropping to **$58.8\,\mathrm{Ah}$** in the critical near-EOL regime ($\le 200\,\mathrm{Ah}$ remaining).

---

## 📁 Repository Structure

```text
├── README.md                            <- Project overview and documentation
├── 24_pack_endpoints.py                 <- NASA ALT dataset audit and cohort stratification
├── 25_survival_rul.py                   <- Kaplan-Meier survival curves and LOPO RUL regression
├── 26_report_tables.py                  <- Report table formatting and summary generation
├── PATCH_v73.md                         <- Patch and calibration notes (v7.3)
│
├── NASA-MultiAgent-BMS/                 <- Core Python BMS package and simulation suite
│   ├── configs/                         <- Calibrated simulation and algorithm YAML configs
│   ├── mabms/                           <- Core library
│   │   ├── agents/                      <- Proposer agents (Speed, Thermal, Health, SingleAgent)
│   │   ├── environment/                 <- Battery simulator (2RC Thevenin ECM + thermal + ageing)
│   │   ├── estimation/                  <- 4-State Adaptive EKF and OCV lookup curves
│   │   ├── evaluation/                  <- Controller benchmarking, tuning, and paired tests
│   │   └── supervisor/                  <- Deterministic analytical safety supervisor
│   ├── scripts/                         <- End-to-end execution pipeline (00_download to 26_report)
│   ├── results/                         <- Raw experimental telemetry CSVs, JSONs, and metrics
│   ├── tests/                           <- Comprehensive test suite (37 passed)
│   └── requirements.txt                 <- Python dependencies
│
└── Thesis_Template___Amrita_AID/        <- Final Year Bachelor's / Master's Thesis (LaTeX)
    ├── mythesis.tex                     <- Main LaTeX document (Amrita AID template)
    ├── mythesis.pdf                     <- Compiled 76-page final thesis document
    ├── mybib.bib                        <- 62 formal academic references
    ├── generate_thesis_graphs.py        <- Generates all 14 publication-grade figures (300 DPI)
    ├── chapters/                        <- Thesis chapters (1 to 8 + Appendix)
    └── images/                          <- 14 high-resolution thesis figures
```

---

## 📊 Embedded Thesis Visualizations

The thesis contains **14 publication-grade figures** (300 DPI) generated from empirical telemetry:

| Chapter | Figure | Description |
| :---: | :---: | :--- |
| **Ch 3** | **Figure 3.1** | Measured discharge capacity retention trajectories across 26 packs |
| **Ch 3** | **Figure 3.2** | Rest-cooling thermal relaxation ($\tau = 745.8\,\mathrm{s}$) and time constant distribution |
| **Ch 3** | **Figure 3.3** | Empirical series resistance growth ($R_0$) vs SOH ($r = 0.54, R_0^{\mathrm{new}} = 0.0342\,\Omega$) |
| **Ch 4** | **Figure 4.1** | Calibrated OCV curves and differential voltage derivative ($dV_{\mathrm{oc}}/d\mathrm{SOC}$) |
| **Ch 4** | **Figure 4.2** | Dynamic terminal voltage tracking (residuals $\pm 18\,\mathrm{mV}$) and cross-pack RMSE |
| **Ch 5** | **Figure 5.1** | Vector architecture block diagram of the closed-loop MA-BMS (TikZ) |
| **Ch 6** | **Figure 6.1** | Controller performance metric distributions on 9 unseen test episodes |
| **Ch 6** | **Figure 6.2** | Dynamic charging profiles: proactive current throttling and thermal containment |
| **Ch 6** | **Figure 6.3** | Dirichlet 10,000-sample objective sensitivity and ternary Pareto simplex |
| **Ch 6** | **Figure 6.4** | Controller robustness under sensor noise scaling ($\times 1, \times 2, \times 4$) |
| **Ch 6** | **Figure 6.5** | Pareto frontier in the (time-to-target, capacity loss) plane |
| **Ch 7** | **Figure 7.1** | Final recorded State of Health distribution across the primary cohort |
| **Ch 7** | **Figure 7.2** | Kaplan--Meier survival curves across Endpoints A, B, and C |
| **Ch 7** | **Figure 7.3** | Leave-One-Pack-Out RUL parity plot and lifecycle stage uncertainty shrinkage |

---

## 🚀 Quick Start & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/Mahizhan-S/NASA-MultiAgent-BMS.git
cd NASA-MultiAgent-BMS
pip install -r NASA-MultiAgent-BMS/requirements.txt
```

### 2. Run Test Suite
```bash
cd NASA-MultiAgent-BMS
pytest tests/ -v
# 37 passed in 16.80s
```

### 3. Re-run Controller Benchmarking & Prognostics
```bash
# In NASA-MultiAgent-BMS root:
export PYTHONPATH=.
python scripts/06_compare_controllers.py 30 3
python scripts/24_pack_endpoints.py
python scripts/25_survival_rul.py
```

### 4. Compile Thesis Document
```bash
cd Thesis_Template___Amrita_AID
# Using Tectonic (XeTeX engine):
tectonic mythesis.tex
```

---

## 🎓 Academic Citation

If you build upon this work, please cite:

```bibtex
@mastersthesis{mahizhan2026mabms,
  title        = {Multi-Agent Battery Management System for Predictive Health Estimation, Thermal Regulation, and Safe Charging Control},
  author       = {Mahizhan, S. and Pair Programmers},
  school       = {Department of Artificial Intelligence, Amrita Vishwa Vidyapeetham},
  year         = {2026},
  address      = {Coimbatore Campus, India},
  note         = {Evaluated on NASA PCoE Accelerated Life Testing Dataset}
}
```

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
