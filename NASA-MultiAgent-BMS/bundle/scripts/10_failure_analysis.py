"""Split deadline misses into near-miss (termination artefact) vs genuine starvation. Needs episodes CSVs produced AFTER the env patch
(columns target_soc, shortfall). Usage: python scripts/10_failure_analysis.py [results/compare_test_unseen_episodes.csv]"""
import sys
import numpy as np, pandas as pd
f = sys.argv[1] if len(sys.argv) > 1 else "results/compare_test_unseen_episodes.csv"
d = pd.read_csv(f)
if "is_starvation" not in d: sys.exit("No 'is_starvation' column: rerun scripts/06 with the patched environment.")
fail = d[~d["target_reached"].astype(bool)]
rows = []
for m, g in d.groupby("method"):
    fg = g[~g["target_reached"].astype(bool)]
    rows.append({"method": m, "episodes": len(g), "failed": len(fg), "fail_rate": len(fg) / len(g),
                 "near_miss_lt_0.01": float((fg.shortfall < 0.01).mean()) if len(fg) else np.nan,
                 "median_shortfall": float(fg.shortfall.median()) if len(fg) else np.nan,
                 "p90_shortfall": float(fg.shortfall.quantile(.9)) if len(fg) else np.nan,
                 "max_shortfall": float(fg.shortfall.max()) if len(fg) else np.nan,
                 "near_miss_rate": float(g.is_near_miss.astype(bool).mean()),
                 "starvation_rate": float(g.is_starvation.astype(bool).mean())})
t = pd.DataFrame(rows); print(t.round(4).to_string(index=False)); t.to_csv(f.replace(".csv", "_failure_breakdown.csv"), index=False)
