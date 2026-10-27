"""Comparative study of SOH / temperature estimators. Requires scripts/02_preprocess.py first."""
import json
from pathlib import Path
import pandas as pd
from mabms.config import load_config
from mabms.data.loader import load_all
from mabms.evaluation.compare_estimators import compare_soh, compare_temperature

cfg = load_config(); res = Path("results"); res.mkdir(exist_ok=True); k = cfg["data"]["split"]["n_cv_folds"]
sp = json.loads(Path(cfg["data"]["manifest_dir"], "pack_split.json").read_text()); dev = sp["train"] + sp["val"]; test = sp["test"]
tab = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "cycle_table.csv")
print(f"dev packs={len(dev)}, test packs={len(test)}  (small n => wide uncertainty; interpret p-values cautiously)")
try:
    models = ("mean", "ridge", "hgb") + (("xgboost",) if __import__("importlib").util.find_spec("xgboost") else ())
    s = compare_soh(tab, dev, test, k, models, cfg["seed"])
    s["cv"].to_csv(res / "cmp_soh_cv.csv", index=False); s["test"].to_csv(res / "cmp_soh_test.csv", index=False); s["pairwise_cv"].to_csv(res / "cmp_soh_pairwise_cv.csv", index=False)
    print("\nSOH grouped-CV (dev):\n", s["cv"].round(4).to_string(index=False), "\nbest by CV:", s["best_by_cv"])
    print("\nSOH held-out TEST packs:\n", s["test"].round(4).to_string(index=False)); print("\nPairwise (per-pack MAE):\n", s["pairwise_cv"].round(4).to_string(index=False))
except Exception as e:
    print("SOH comparison failed:", repr(e))
packs = load_all(cfg["data"]["raw_dir"], cfg)
t = compare_temperature(packs, dev, test, cfg["temperature"], k, cfg["seed"])
t["cv"].to_csv(res / "cmp_temp_cv.csv", index=False); t["pairwise_cv"].to_csv(res / "cmp_temp_pairwise_cv.csv", index=False)
print("\nTemperature grouped-CV (dev):\n", t["cv"].round(4).to_string(index=False))
if "test" in t: t["test"].to_csv(res / "cmp_temp_test.csv", index=False); print("\nTemperature TEST packs:\n", t["test"].round(4).to_string(index=False))
