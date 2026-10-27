"""Fit the empirical degradation law on TRAIN packs, choose variant on VAL packs, bootstrap CIs; TEST only with --final-test.
Run AFTER 13_calibrate_sim.py with MABMS_CONFIG=configs/calibrated.yaml.  Writes configs/degradation_patch.yaml."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
from mabms.config import load_config
from mabms.data.loader import load_all
from mabms.environment import degradation_fit as dg
from mabms.environment.calibration import VoltParams
from mabms.estimation.soc_estimator import OCVCurve

cfg = load_config(); res = Path("results"); final = "--final-test" in sys.argv
sp = json.loads(Path(cfg["data"]["manifest_dir"], "pack_split.json").read_text()); s = cfg["simulator"]
ct_all = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "cycle_table.csv"); ocv = OCVCurve(s["ocv_soc"], s["ocv_v"])
vp = VoltParams(s["r0_ohm"], s["r_ea_k"], s["r1_ohm"], max(30.0, s["r1_ohm"] * s["c1_f"]))
ids = sp["train"] + sp["val"] + (sp["test"] if final else []); raw = load_all(cfg["data"]["raw_dir"], cfg)
P = {}
for pid in ids:
    ct = ct_all[ct_all.pack_id == pid].sort_values("t_start").reset_index(drop=True)
    if len(ct) >= 4 and pid in raw and ct.cap_ref_ah.median() >= 2.0:
        ct_use = ct.copy(); ct_use["soh"] = ct_use.capacity_ah / s["q0_ah"]
        P[pid] = dg.prepare_pack(raw[pid], ct_use, cfg, ocv, vp, s["q0_ah"])
tr = [p for p in sp["train"] if p in P]; va = [p for p in sp["val"] if p in P]
print(f"packs with >=4 reference checks: train={len(tr)} val={len(va)}")
rows, fits = [], {}
for v in dg.VARIANTS:
    par = dg.fit_variant([P[p] for p in tr], v); fits[v] = par; e = dg.evaluate([P[p] for p in va], va, par)
    rows.append({"variant": v, **par, "val_mae_soh": e.mae_soh.mean(), "val_rmse_soh": float(np.sqrt((e.rmse_soh ** 2).mean())), "val_end_err": e.end_err_soh.abs().mean()})
a = cfg["simulator"]["degradation"]; assumed = {"k": a["k_per_eq_cycle"], "ea": a["ea_over_r_k"], "a": a["c_rate_coeff"], "b": a["soc_stress_coeff"]}
e = dg.evaluate([P[p] for p in va], va, assumed); rows.append({"variant": "ASSUMED_placeholder", **assumed, "val_mae_soh": e.mae_soh.mean(),
        "val_rmse_soh": float(np.sqrt((e.rmse_soh ** 2).mean())), "val_end_err": e.end_err_soh.abs().mean()})
tab = pd.DataFrame(rows); tab.to_csv(res / "degradation_variants_val.csv", index=False); print(tab.round(5).to_string(index=False))
best = tab[tab.variant.isin(dg.VARIANTS)].sort_values("val_mae_soh").iloc[0]["variant"]; print("\nselected on VAL:", best)
bs = dg.bootstrap_params([P[p] for p in tr], best); ci = bs.quantile([0.025, 0.5, 0.975]).round(5); ci.to_csv(res / "degradation_bootstrap_ci.csv"); print("pack-bootstrap 95% CI:\n", ci.to_string())
dg.evaluate([P[p] for p in va], va, fits[best]).to_csv(res / "degradation_val_per_pack.csv", index=False)
if final:
    te = [p for p in sp["test"] if p in P]; et = dg.evaluate([P[p] for p in te], te, fits[best]); et.to_csv(res / "degradation_test_per_pack.csv", index=False)
    print("\nTEST (reported once): MAE SOH =", round(et.mae_soh.mean(), 5), " per pack:\n", et.round(4).to_string(index=False))
f = {k: float(v) for k, v in fits[best].items()}
Path("configs/degradation_patch.yaml").write_text(yaml.safe_dump({"simulator": {"degradation": {"k_per_eq_cycle": f["k"], "ea_over_r_k": f["ea"], "c_rate_coeff": f["a"], "soc_stress_coeff": f["b"]}}}))
print("Wrote configs/degradation_patch.yaml (merge into configs/calibrated.yaml under simulator.degradation)")
