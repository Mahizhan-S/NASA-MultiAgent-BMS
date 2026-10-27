"""What do the RAW files contain during rest (mode 0) and charge (mode 1)? Is temperature logged there? What is the rest-cooling time constant?
Usage: python scripts/18_mode_audit.py   (reads data.raw_dir; writes results/mode_audit_*.csv)"""
from pathlib import Path
import pandas as pd
from mabms.config import load_config
from mabms.data.loader import pack_id_from_path
from mabms.data.mode_audit import audit_file

cfg = load_config(); root = Path(cfg["data"]["raw_dir"]); res = Path("results"); res.mkdir(exist_ok=True)
M, B, C = [], [], []
for p in sorted(root.rglob("*.csv")):
    pid = pack_id_from_path(p, root, cfg["data"].get("pack_id_regex")); a = audit_file(p, pid)
    M.append(a["modes"]); B.append(a["blocks"]); C.append(a["cooling"]); print("done", pid, flush=True)
M, B, C = pd.concat(M), pd.concat(B), pd.concat(C)
M.to_csv(res / "mode_audit_columns.csv", index=False); B.to_csv(res / "mode_audit_blocks.csv", index=False); C.to_csv(res / "mode_audit_rest_cooling.csv", index=False)
nan_cols = [c for c in M.columns if c.startswith("nan_")]
print("\nMean NaN fraction of each column, by mode (0 = always populated, 1 = never):")
print(M.groupby("mode")[nan_cols + ["share"]].mean().round(3).to_string())
print("\nBlock durations by mode (median over packs):")
print(B.groupby("mode")[["n_blocks", "median_dur_s", "p10_dur_s", "p90_dur_s", "median_dT"]].median().round(1).to_string())
if len(C):
    print(f"\nRest-block cooling fits: {len(C)} blocks from {C.pack_id.nunique()} packs; tau_s median {C.tau_s.median():.0f}  IQR [{C.tau_s.quantile(.25):.0f}, {C.tau_s.quantile(.75):.0f}]")
    print("per-pack median tau_s:\n", C.groupby("pack_id").tau_s.median().round(0).describe().round(0).to_string())
    print("\nCompare with the calibrated model: tau = C_th / h =", end=" ")
    s = cfg["simulator"]["thermal"]; print(round(s["c_th_j_per_k"] / s["h_w_per_k"]), "s")
else:
    print("\nNo usable rest blocks (temperature missing or too short).")
