"""v7.3 part 3: weight-sensitivity + Pareto from stored episodes (no re-simulation).
Edit COLS to match your episode CSV (script prints the available columns if a name is missing).
Usage: python scripts/26_report_tables.py results/episodes_06.csv
"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COLS = dict(method="method", time="time_to_target_s", deg="cap_loss", cool="cool_wh", success="target_reached")
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/episodes_06.csv")
miss = [v for v in COLS.values() if v not in df]
if miss: sys.exit(f"missing columns {miss}; available: {list(df.columns)}")
g = df.groupby(COLS["method"]).agg(tt=(COLS["time"], "mean"), dd=(COLS["deg"], "mean"),
                                   ee=(COLS["cool"], "mean"), ss=(COLS["success"], "mean"))
print("RAW METRICS (report these, not J alone):\n", g.round(4).to_string())
sc = g[["tt", "dd", "ee"]].max()           # normalise by max across methods
Z = g[["tt", "dd", "ee"]] / sc
rng = np.random.default_rng(0); W = rng.dirichlet(np.ones(3), 5000)
J = W @ Z.T.values                       # (5000, methods)
best = pd.Series(np.bincount(J.argmin(axis=1), minlength=len(g)) / len(W), index=g.index)
base = (Z.sum(axis=1)).rank().values
tau = [pd.Series(j).rank().corr(pd.Series(base), method="kendall") for j in J]
print("\nShare of 5000 random weightings where each method is best:\n", best.round(3).to_string())
print("Median Kendall tau of ranking vs equal weights:", round(float(np.median(tau)), 2))
nd = [m for m in g.index if not ((g.drop(m).tt <= g.tt[m]) & (g.drop(m).dd <= g.dd[m]) & (g.drop(m).ee <= g.ee[m])).any()]
print("Pareto-nondominated methods (T,D,E all minimised):", nd)
fig, ax = plt.subplots()
for m in g.index:
    ax.scatter(g.tt[m], g.dd[m], s=200 * Z.ee[m] + 20, label=m, edgecolor="k" if m in nd else "none")
ax.set_xlabel("mean time to target (s)"); ax.set_ylabel("mean capacity loss"); ax.legend(fontsize=7)
fig.savefig("results/pareto_v73.png", dpi=150, bbox_inches="tight")
g.assign(best_share=best).to_csv("results/weight_sensitivity_v73.csv")
