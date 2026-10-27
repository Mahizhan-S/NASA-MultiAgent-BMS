# Patch v7: raw-data checks for the charge protocol and the end-of-life definition
Apply on top of bundle2 state (v4 + v5/v6). Files: mabms/data/mode_audit.py (superset of the v5 version), scripts/21_charge_protocol_checks.py, tests/test_charge_checks.py

## Run (reads the RAW csvs; a few minutes)
export PYTHONPATH=. MABMS_CONFIG=configs/calibrated.yaml
python scripts/21_charge_protocol_checks.py 2>&1 | tee logs/21.txt
Send: logs/21.txt, results/charge_block_checks.csv, results/end_of_data_checks.csv

## What each part tests
(A) charge-phase heating amplitude A = P/h per charge block (model-free of the charge current): the claim "~5.4 C rise" predicts A ~ 5.4 C.
(B) voltage_charger plateau fraction: a long plateau at the block maximum suggests a voltage-limited (CV-like) phase; a short one suggests pure CC.
(C) end of data per pack: abrupt collapse (rel_dur << 1 or very low v_min) means the pack FAILED (paper: EOL = cell failure, typically a short circuit),
    which would make the 15 packs that never reach SOH<=0.8 failure EVENTS, not right-censored survivors.

## Config suggestion (not required by the scripts)
agents.conventional.c_rate: 1.2     # the dataset's own charging policy is a constant 3 A (= 1.2 C of 2.5 Ah)
