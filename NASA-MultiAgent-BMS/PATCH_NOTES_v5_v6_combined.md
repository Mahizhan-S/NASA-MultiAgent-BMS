# Combined patch (v5 + v6) - apply ONCE on top of your current state (bundle + patch_v4)

## 1. Apply
unzip -o patch_v5_v6_combined.zip -d <your project root>        # overwrites only the files listed below
Do NOT apply patch_v5.zip or patch_v6.zip separately (this zip contains both; stress.py is the v6 superset).

## 2. Merge this block into configs/calibrated.yaml  (and into configs/default.yaml if you run tests against it)
```yaml
estimator:
  adaptive_capacity: true
  adapt_r0: true            # 4-state EKF [soc, v1, capacity ratio, resistance ratio]
  p0_rho: 0.09
  q_rho: 1.0e-8
safety:
  v_margin: 0.02
  r0_margin: 1.3
  temp_monitor: {enabled: true, thr_c: 4.0, k: 3, gain: 0.002}
agents:
  charging: {soc_stop_margin: 0.0, restart_hysteresis: 0.0}      # keep the defaults; the stop-latch was tested and rejected
simulator:
  degradation:              # k_T variant (identifiable) from your results/degradation_variants_val.csv
    k_per_eq_cycle: 1.7654e-4
    ea_over_r_k: 4963.7
    c_rate_coeff: 0.0
    soc_stress_coeff: 0.0
objective:
  d_scale: auto
  e_scale: auto
limits:
  i_max_a: 4.9
```
(Only ADD or CHANGE these keys; keep everything else in your calibrated.yaml. If a key already exists, overwrite its value.)

## 3. Run, in this order (export once)
export PYTHONPATH=. MABMS_CONFIG=configs/calibrated.yaml; mkdir -p logs
pytest -q tests                                            2>&1 | tee logs/pytest.txt
python scripts/18_mode_audit.py                            2>&1 | tee logs/18.txt      # reads RAW files, takes a few minutes
python scripts/06_compare_controllers.py 30 3              2>&1 | tee logs/06.txt      # re-tunes the single agent under the new config
python scripts/10_failure_analysis.py                      2>&1 | tee logs/10.txt
python scripts/19_estimator_stop_sweep.py 60 2             2>&1 | tee logs/19.txt
python scripts/17_rq3_stress.py 30                         2>&1 | tee logs/17.txt
python scripts/20_temp_monitor_eval.py 20                  2>&1 | tee logs/20.txt
python scripts/07_ablations_robustness.py 20 2             2>&1 | tee logs/07.txt
(Run 06 BEFORE 17 and 20: they read results/single_agent_best_params.json.)

## 4. Contents
new: mabms/data/mode_audit.py, mabms/evaluation/stress.py (v6 superset), scripts/17_rq3_stress.py, scripts/18_mode_audit.py,
     scripts/19_estimator_stop_sweep.py, scripts/20_temp_monitor_eval.py, tests/test_stress.py, tests/test_mode_audit.py, tests/test_v6.py
changed: mabms/agents/{base,charging_agent,single_agent,safety_agent}.py, mabms/estimation/soc_estimator.py, mabms/evaluation/run_independent.py

## 5. Notes
- All simulator results (17, 19, 20, 06, 07) are simulation outputs, not NASA measurements.
- Script 18 reads the raw CSVs only; it does not need the loader.
- Do not run --final-test yet.
