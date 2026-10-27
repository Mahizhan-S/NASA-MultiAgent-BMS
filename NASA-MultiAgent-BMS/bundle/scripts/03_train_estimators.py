"""Phase 2: grouped-CV on TRAIN+VAL packs only (test packs untouched). Saves metrics to results/."""
import json
from pathlib import Path
import pandas as pd
from mabms.config import load_config
from mabms.data.loader import load_all
from mabms.data.segments import build_cycle_table
from mabms.estimation import soh_estimator as soh, temperature_predictor as tp, rul_predictor as rul

cfg = load_config(); res = Path("results"); res.mkdir(exist_ok=True)
split = json.loads(Path(cfg["data"]["manifest_dir"], "pack_split.json").read_text())
dev = set(split["train"]) | set(split["val"]); k = cfg["data"]["split"]["n_cv_folds"]
tab = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "cycle_table.csv"); tab = tab[tab.pack_id.isin(dev)]
print("dev packs:", tab.pack_id.nunique(), "cycles:", len(tab))
out = {}
for m in ["mean", "ridge", "hgb"]:
    r = soh.grouped_cv(tab, m, k, seed=cfg["seed"]); out[f"soh_{m}"] = r["overall"]
    r["per_pack"].to_csv(res / f"soh_{m}_per_pack.csv", index=False); print("SOH", m, r["overall"])
packs = {p: d for p, d in load_all(cfg["data"]["raw_dir"], cfg).items() if p in dev}
T = cfg["temperature"]
tr = tp.grouped_cv(packs, T["lags"], T["horizon_steps"], T["stride"], k, cfg["seed"])
for m, r in tr.items():
    out[f"temp_{m}"] = r["overall"]; r["per_pack"].to_csv(res / f"temp_{m}_per_pack.csv", index=False); print("TEMP", m, r["overall"])
cyc = {p: g for p, g in tab.groupby("pack_id")}
rt, cens = rul.build_rul_table(cyc, cfg["rul"]["soh_eol"], cfg["rul"]["min_cycles_per_pack"])
print(f"RUL: {rt.pack_id.nunique() if len(rt) else 0} packs reach EOL (SOH<={cfg['rul']['soh_eol']}); censored: {len(cens)}")
cens.to_csv(res / "rul_censored_packs.csv", index=False)
if len(rt) and rt.pack_id.nunique() >= 3:
    for m, r in rul.grouped_cv(rt, k, cfg["seed"]).items():
        out[f"rul_{m}"] = r["overall"]; print("RUL", m, r["overall"])
else:
    print("Too few EOL packs for a defensible RUL model: SKIPPED (documented limitation).")
(res / "estimator_metrics_dev.json").write_text(json.dumps(out, indent=2))
