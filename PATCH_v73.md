# Patch v7.3 - changes required BEFORE writing the report

## A. Run order
```
python scripts/24_pack_endpoints.py 2>&1 | tee logs/24.txt          # add --basis rel for the sensitivity run
python scripts/25_survival_rul.py   2>&1 | tee logs/25.txt
# (after simulator change C and re-runs of 06/10/17)
python scripts/26_report_tables.py results/<your episodes csv> 2>&1 | tee logs/26.txt
```
Send back: logs/24.txt, logs/25.txt, results/pack_endpoints_v73.csv, results/rul_lopo_v73.csv, logs/26.txt.

## B. Data/label decisions (implemented in 24)
1. Two cohorts. cap_ref < 2.0 Ah  => "recommissioned" (packs 13, 36, 54). Main analysis = 23 original packs. Recommissioned packs reported separately; never pooled into the 0.8-SOH threshold analysis.
2. SOH basis = capacity / 2.45 Ah (absolute). Run once with --basis rel as sensitivity; report both.
3. Threshold crossing = first reference with SOH<=0.8 that is confirmed by the next reference (or is the last one). Packs 22 (truncated last discharge) and 41 (crosses, recovers, fails) are flagged.
4. Failure time is interval-censored: [last ref throughput, +60 Ah].
5. Packs with <=7 reference checks are excluded from regression RUL, kept in survival.
6. Labels: event_strong / event_probable / event_charger_dead = failure; transient_collapse (54) and no_failure_evidence = censored.

## C. Simulator changes (apply by hand; I cannot see these files)
1. battery_env.make_scenarios(): draw initial SOH  soh0 ~ U(0.75, 1.0) per episode, set Q = q0_ah * soh0 and self.soh = soh0, so r0_scale(soh0) applies from step 0 (recommissioned packs start at ~0.76-0.77 absolute).
2. soc_estimator.make_soc_estimator(): rho0 = r0_scale(soh_est) must use the same absolute SOH.
3. Keep k_R0 ~ U(1.5, 4.5). Do not describe 3.0 as calibrated: fit r=0.54 across 26 packs.
4. After 1-3: re-run scripts 06, 10, 17 and 14 (degradation fit must use absolute SOH for the original cohort).

## D. Claims to change in the report
- RQ1 (health cap reduces degradation): not supported. Capacity-loss differences are within noise; independent_health_cap success fell from 93.3% to 65.6% after R0(SOH) was added, i.e. it is fragile.
- J ranking: report raw metrics (time, capacity loss, cooling, success) plus the weight-sensitivity table; do not present scalar J as the headline.
- Conventional-charger failures went 3.3% -> 0%: that depends on baseline c_rate (1.2 = 2.94 A, matching the dataset's 3 A) and on the R0 law. Present as a sensitivity result, not a controller finding.
- Charge physics: only the thermal response is validated against charge data. Charge current/voltage dynamics and impedance are NOT validated (no charge current logged). Remove "physical validation against charging telemetry" from the literature matrix, or restrict it to thermal.
- Safety: constant bias is undetectable by the temperature monitor; cost of safety is lower success under tight deadlines. State both.
- EOL: 15 of 26 packs ended in failure; 10 of those were still above 0.8 SOH at their last reference check, so SOH-based RUL cannot predict them. No precursor found in heating amplitude. Say this explicitly.
- Charging protocol: CC at about 3 A with termination at the voltage limit (plateau fraction 2-8%); current is inferred from the paper, not logged. Some blocks hit a 3599 s time cap in packs 50-53.
- Pack 54: tau about 300 s vs 560-914 s elsewhere; excluded from thermal calibration statistics.

## E. Decisions only you can make
1. Coordinator / RQ2: either (a) build a minimal constrained arbiter (needs the agents' propose() interface; send base.py), or (b) drop RQ2 and state it as future work / not tested. Do not claim RQ2 without it.
2. Safety margin sweep: needs script 17 to accept a --config path (or tell me how it is parameterised).
3. pack_split.json: send it so test-set RUL/SOH tables can be reported on the 5 reserved packs.

## F. Reproducibility
Fresh-clone rerun of 00-26, pin package versions, README with run order, record c_rate=1.2 and k_R0 range in the config, commit bundle hashes.
