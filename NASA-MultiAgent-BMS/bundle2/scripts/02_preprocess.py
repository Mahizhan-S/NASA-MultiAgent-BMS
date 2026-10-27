"""Load -> clean -> segment -> REFERENCE-ONLY cycle table (throughput axis) -> pack-level split manifest.
The existing data/manifests/pack_split.json is REUSED (test packs stay fixed). Use --resplit only before looking at any test results."""
import json, sys, warnings
from pathlib import Path
import pandas as pd
from mabms.config import load_config
from mabms.data.loader import load_all
from mabms.data.segments import build_cycle_table
from mabms.data.splits import split_packs

cfg = load_config(); root = Path(cfg["data"]["raw_dir"]); resplit = "--resplit" in sys.argv
packs = load_all(root, cfg)
print("packs loaded:", len(packs))
tables = {}
for pid, df in packs.items():
    with warnings.catch_warnings():
        warnings.simplefilter("error" if "mission_type" in df.columns else "default")
        t = build_cycle_table(df, cfg)
    print(f"{pid}: rows={len(df)} reference checks={len(t)}" + ("" if len(t) else "   <-- NO reference cycles (check thresholds)"))
    if len(t): tables[pid] = t
out = Path(cfg["data"]["processed_dir"]); out.mkdir(parents=True, exist_ok=True)
pd.concat(tables.values(), ignore_index=True).to_csv(out / "cycle_table.csv", index=False)
sp_path = Path(cfg["data"]["manifest_dir"], "pack_split.json")
if sp_path.exists() and not resplit:
    sp = json.loads(sp_path.read_text()); known = set().union(*sp.values())
    missing = known - set(tables)
    print("REUSING existing split:", {k: len(v) for k, v in sp.items()}, "| packs without reference cycles:", sorted(missing) or "none")
else:
    sp = split_packs(list(tables), cfg["data"]["split"]["test_fraction"], cfg["data"]["split"]["val_fraction"], cfg["seed"])
    sp_path.write_text(json.dumps(sp, indent=2)); print("NEW split:", {k: len(v) for k, v in sp.items()})
