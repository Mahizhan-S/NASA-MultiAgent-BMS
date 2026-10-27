"""Simulator calibration (OCV, voltage params with tau>=30 s, thermal) on TRAIN packs; validation on VAL; TEST only with --final-test.
Writes results/sim_calibration.json, results/sim_val_*.csv and configs/calibrated.yaml.  Usage: python scripts/13_calibrate_sim.py [--final-test]"""
import json, sys
from pathlib import Path
import pandas as pd, yaml
from mabms.config import load_config
from mabms.data.extra import attach_extra_column
from mabms.data.loader import load_all
from mabms.environment import calibration as cal
from mabms.estimation.soc_estimator import OCVCurve

cfg = load_config(); res = Path("results"); res.mkdir(exist_ok=True); final = "--final-test" in sys.argv
sp = json.loads(Path(cfg["data"]["manifest_dir"], "pack_split.json").read_text())
ct_all = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "cycle_table.csv")
use = sp["train"] + sp["val"] + (sp["test"] if final else [])
packs = {p: d for p, d in load_all(cfg["data"]["raw_dir"], cfg).items() if p in use}
CHG = ["voltage_charger"]
def segs_for(ids, n_per_type):
    out, amb = [], {}
    for pid in ids:
        ct = ct_all[ct_all.pack_id == pid].sort_values("t_start")
        if len(ct) < 2: continue
        df = attach_extra_column(packs[pid], cfg["data"]["raw_dir"], cfg, CHG)
        out += cal.extract_segments(df, ct, cfg, n_per_type=n_per_type); amb[pid] = cal.ambient_proxy(packs[pid])
    return out, amb
train, amb_tr = segs_for(sp["train"], 6); val, amb_va = segs_for(sp["val"], 6)
print(f"train segments={len(train)}  val segments={len(val)}  (charger-voltage charge segs: train={sum(s['v_is_charger'] for s in train)})")
s0 = cfg["simulator"]; p = cal.VoltParams(s0["r0_ohm"], s0["r_ea_k"], s0["r1_ohm"], max(cal.TAU_MIN_S, s0["r1_ohm"] * s0["c1_f"]))
ocv0 = OCVCurve(s0["ocv_soc"], s0["ocv_v"])
before = cal.validate_voltage(val, p, ocv0)                                  # PLACEHOLDER model as reference
ocv_d = {"soc": s0["ocv_soc"], "v": s0["ocv_v"]}; ocv = ocv0
for it in range(2):                                                          # alternate OCV <-> R estimates
    ocv_d = cal.fit_ocv_from_reference(train, p); ocv = OCVCurve(ocv_d["soc"], ocv_d["v"]); p = cal.fit_voltage_params(train, ocv, p)
    print(f"iter {it}: {p}  tau={p.tau_s:.1f}s  C1={p.c1:.1f}F")
c_th, h = cal.fit_thermal(train, p, amb_tr); print(f"thermal: C_th={c_th:.2f} J/K  h={h:.4f} W/K")
after = cal.validate_voltage(val, p, ocv); vt = cal.validate_temperature(val, p, c_th, h, amb_va)
for name, df in [("before_placeholder", before), ("after", after)]:
    print(f"\nVAL voltage RMSE [{name}] by kind:\n", df.groupby("kind")["rmse_v"].agg(["mean", "median", "max"]).round(4).to_string())
print("\nVAL temperature RMSE by kind:\n", vt.groupby("kind")["rmse_c"].agg(["mean", "median", "max"]).round(3).to_string())
chg = after[after.kind == "charge"]
if len(chg): print("\nCharge segments use charger voltage (incl. cable drop) for:", int(chg.v_is_charger.sum()), "of", len(chg))
before.to_csv(res / "sim_val_voltage_before.csv", index=False); after.to_csv(res / "sim_val_voltage_after.csv", index=False); vt.to_csv(res / "sim_val_temperature.csv", index=False)
out = {"volt_params": p.__dict__ | {"c1_f": p.c1}, "thermal": {"c_th": c_th, "h": h}, "ocv": ocv_d, "ambient_proxy": "5th percentile of pack temperature",
       "val_voltage_rmse_after_mean": float(after.rmse_v.mean()), "val_voltage_rmse_before_mean": float(before.rmse_v.mean())}
if final:
    test, amb_te = segs_for(sp["test"], 6); tv = cal.validate_voltage(test, p, ocv); tv.to_csv(res / "sim_test_voltage.csv", index=False)
    print("\nTEST voltage RMSE (reported once):\n", tv.groupby("kind")["rmse_v"].agg(["mean", "median", "max"]).round(4).to_string()); out["test_voltage_rmse_mean"] = float(tv.rmse_v.mean())
(res / "sim_calibration.json").write_text(json.dumps(out, indent=2))
Path("configs/calibrated.yaml").write_text(yaml.safe_dump(cal.calibrated_config(cfg, p, ocv_d, c_th, h), sort_keys=False))
print("\nWrote results/sim_calibration.json and configs/calibrated.yaml  (use: export MABMS_CONFIG=configs/calibrated.yaml)")
