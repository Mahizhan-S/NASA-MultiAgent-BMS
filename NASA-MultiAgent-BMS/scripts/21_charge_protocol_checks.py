"""Checks that need the RAW files (paper: charge = constant 3 A, 10 min rests, EOL = cell failure):
 (A) charge-phase heating: amplitude A = P/h per charge block (compare with the I^2*R0/h prediction, ~5.4 C per cell if R0=0.041 and h=0.068),
 (B) voltage_charger plateau fraction: a long plateau at the maximum suggests a voltage-limited (CV-like) phase,
 (C) end of data: does each pack file end in an abrupt collapse (failure) or just stop?  -> decides whether 'censored' packs are really failures.
Usage: python scripts/21_charge_protocol_checks.py"""
from pathlib import Path
import numpy as np, pandas as pd
from mabms.config import load_config
from mabms.data.loader import pack_id_from_path
from mabms.data.mode_audit import charge_block_stats, end_of_data, load_raw, rest_tau_ta

cfg = load_config(); root = Path(cfg["data"]["raw_dir"]); res = Path("results"); res.mkdir(exist_ok=True)
CB, EO = [], []
for p in sorted(root.rglob("*.csv")):
    pid = pack_id_from_path(p, root, cfg["data"].get("pack_id_regex")); d = load_raw(p); tau, ta = rest_tau_ta(d)
    if np.isfinite(tau):
        cb = charge_block_stats(d, tau, ta); cb["pack_id"] = pid; cb["tau"] = tau; cb["Ta"] = ta; CB.append(cb)
    e = end_of_data(d)
    if e:
        x = e["last_discharges"]; EO.append({"pack_id": pid, "baseline_dur_s": e["baseline_dur_s"], "last_rel_dur": x.rel_dur.iloc[-1], "min_rel_dur_last6": x.rel_dur.min(),
                  "v_first_last": x.v_first.iloc[-1], "v_min_last": x.v_min.iloc[-1], "v_min_min_last6": x.v_min.min(), "last_charge_rel_dur": e["last_charge_rel_dur"]})
    print("done", pid, flush=True)
CB = pd.concat(CB, ignore_index=True); EO = pd.DataFrame(EO); CB.to_csv(res / "charge_block_checks.csv", index=False); EO.to_csv(res / "end_of_data_checks.csv", index=False)
g = CB.groupby("pack_id").agg(n=("A_heating_c", "size"), A_median=("A_heating_c", "median"), dT_obs=("dT_obs", "median"), plateau=("plateau_frac", "median"), vc_max=("vc_max", "median"))
print("\n(A,B) per-pack charge-phase heating amplitude A (C), observed dT over the block (C), voltage_charger plateau fraction, vc_max (V):\n", g.round(2).to_string())
print("\nOverall: A median %.2f C (IQR %.2f..%.2f);  plateau fraction median %.2f" % (CB.A_heating_c.median(), CB.A_heating_c.quantile(.25), CB.A_heating_c.quantile(.75), CB.plateau_frac.median()))
cc = cfg["simulator"]["thermal"]; print("prediction to compare: A = I^2*R0/h = 3.0^2 *", cfg["simulator"]["r0_ohm"], "/", cc["h_w_per_k"], "=", round(9 * cfg["simulator"]["r0_ohm"] / cc["h_w_per_k"], 2), "C (per-cell-equivalent model)")
print("\n(C) end of data per pack (rel_dur = last discharge duration / median of the 30 before; v_min = lowest pack voltage in the last discharge):\n", EO.round(3).to_string(index=False))
print("\nPacks whose last discharge is abnormal (rel_dur < 0.7 or v_min < 4.0 V):", EO[(EO.last_rel_dur < 0.7) | (EO.v_min_last < 4.0)].pack_id.tolist())
