"""Temperature prediction at several horizons (config temperature.horizons, in samples; ~1 Hz => seconds). Reports ABSOLUTE and DELTA skill vs persistence."""
import json
from pathlib import Path
import pandas as pd
from mabms.config import load_config
from mabms.data.loader import load_all
from mabms.evaluation.compare_estimators import compare_temperature

cfg = load_config(); k = cfg["data"]["split"]["n_cv_folds"]; res = Path("results"); res.mkdir(exist_ok=True)
sp = json.loads(Path(cfg["data"]["manifest_dir"], "pack_split.json").read_text()); dev, test = sp["train"] + sp["val"], sp["test"]
packs = load_all(cfg["data"]["raw_dir"], cfg); T0 = cfg["temperature"]
rows_cv, rows_te = [], []
for h in T0.get("horizons", [60, 300]):
    T = {**T0, "horizon_steps": h, "stride": max(T0.get("stride", 5), 30)}
    o = compare_temperature(packs, dev, test, T, k, cfg["seed"])
    rows_cv.append(o["cv"].assign(horizon_steps=h))
    if "test" in o: rows_te.append(o["test"].assign(horizon_steps=h))
    print(f"\n--- horizon {h} samples: grouped-CV (dev) ---\n", o["cv"].round(4).to_string(index=False))
    if "test" in o: print("--- held-out test packs ---\n", o["test"].round(4).to_string(index=False))
pd.concat(rows_cv).to_csv(res / "temp_horizons_cv.csv", index=False)
if rows_te: pd.concat(rows_te).to_csv(res / "temp_horizons_test.csv", index=False)
print("\nNote: delta_r2 > 0 means skill beyond persistence; absolute R2 is inflated by autocorrelation.")
