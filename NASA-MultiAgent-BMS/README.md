# NASA-MultiAgent-BMS (stage 1: audit -> estimators -> simulator -> agents; NO coordinator yet)

## Run order (from project root; `export PYTHONPATH=.`)
pip install -r requirements.txt
python scripts/00_download.py            # official zip (needs internet)
python scripts/01_data_audit.py          # READ the output, then edit data.* in configs/default.yaml
python scripts/02_preprocess.py          # cycle table (derived capacity/SOH) + pack-level split manifest
python scripts/03_train_estimators.py    # grouped CV on train+val packs only
python scripts/04_validate_sim.py        # fit/validate surrogate on measured discharge (train fit, val report)
python scripts/05_run_independent.py 20  # conventional vs independent agents (simulator only)
pytest -q tests

## Status
- Executed by me: unit tests on SYNTHETIC data and the simulator benchmark (scripts/05). Nothing was run on NASA data (no network in my sandbox).
- UNVERIFIED assumptions to confirm from the audit: column names, current sign, time units, pack<->file mapping,
  full-discharge voltage window (per-cell-equivalent), series/parallel pack layout, nominal capacity.
- Simulator parameters are placeholders; degradation law is a surrogate, NOT validated. SOH in simulation is noisy truth (placeholder).
- Safety agent is a research-simulation component, not a certified BMS.

## Comparative studies (added)
python scripts/06_compare_controllers.py 30 3   # conventional vs single-agent (tuned on train/val) vs independent agents; paired stats + boxplots
python scripts/07_ablations_robustness.py 20 2  # no-health / no-thermal / safety-off (sim-only) + sensor-noise sweep
python scripts/08_compare_estimators.py         # SOH (mean/ridge/HGB/XGB, all vs early-window features), temperature, on dev CV then reserved test packs
Notes: pairing unit = scenario (controllers) or pack (estimators); seeds are averaged within a unit, not treated as independent samples.
p-values use Wilcoxon signed-rank + Holm correction; with few scenarios/packs they have little power. Smoke runs (n=6) are NOT results.
Single-agent baseline = parametric policy + random search on shared objective J (no RL yet). J weights are a design choice (configs: objective).
The coordinator-based method will plug into METHODS in run_independent.py when implemented.
