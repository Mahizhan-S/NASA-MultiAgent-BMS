"""RUL on a throughput axis with fair baselines; EOL packs and censored packs reported SEPARATELY. Requires 02_preprocess.py (v2) first."""
import json
from pathlib import Path
import pandas as pd
from mabms.config import load_config
from mabms.estimation import rul_predictor as rul

cfg = load_config(); res = Path("results"); res.mkdir(exist_ok=True); eol = cfg["rul"]["soh_eol"]
sp = json.loads(Path(cfg["data"]["manifest_dir"], "pack_split.json").read_text())
tab = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "cycle_table.csv")
for name, ids in [("dev", sp["train"] + sp["val"]), ("test", sp["test"])]:
    sub = tab[tab.pack_id.isin(ids)]
    tbl, cens = rul.build_rul_table_v2({p: g for p, g in sub.groupby("pack_id")}, eol, min_refs=6)
    n_eol = tbl.pack_id.nunique() if len(tbl) else 0
    print(f"[{name}] packs total={len(ids)}  reached EOL={n_eol}  censored/too-few={len(cens)}")
    cens.to_csv(res / f"rul_v2_{name}_censored.csv", index=False); tbl.to_csv(res / f"rul_v2_{name}_table.csv", index=False)
    if name == "dev" and n_eol >= 3:
        out = rul.rul_study_v2(tbl, {p: g for p, g in sub.groupby("pack_id")}, cens, eol, cfg["data"]["split"]["n_cv_folds"], cfg["seed"])
        out["eol_packs"].to_csv(res / "rul_v2_dev_eol_metrics.csv", index=False); out["censored_check"].to_csv(res / "rul_v2_dev_censored_check.csv", index=False)
        print("\nDev, EOL packs (RUL in Ah of throughput; leave-pack-group-out):\n", out["eol_packs"].round(2).to_string(index=False))
        cc = out["censored_check"]
        if len(cc): print("\nCensored packs: lower-bound violation rate by model:\n", cc.groupby("model")["violates_lower_bound"].mean().round(3).to_string())
        for k, v in out["per_pack"].items(): v.to_csv(res / f"rul_v2_dev_per_pack_{k}.csv", index=False)
