"""v7.1 part 1: clean charge-block table, per-pack heating-vs-age, charge-duration capacity proxy.
Input : results/charge_block_checks.csv (from script 21)
Output: results/charge_block_filtered.csv, charge_A_by_pack.csv, charge_capacity_proxy.csv
"""
import numpy as np, pandas as pd, sys
R = sys.argv[1] if len(sys.argv) > 1 else "results"
df = pd.read_csv(f"{R}/charge_block_checks.csv")
df["block_idx"] = df.groupby("pack_id").cumcount()
df["time_capped"] = df.duration_s >= 3590
ok = ((df.vc_max >= 8.4) & df.T0.between(15, 70) & (df.dT_obs.abs() < 40)
      & (df.duration_s >= 1000) & df.A_heating_c.between(0, 25))
f = df[ok].copy()
print(f"kept {len(f)}/{len(df)} blocks; dropped {len(df)-len(f)} (rest blocks, glitches, truncated)")
print("dropped per pack:\n", df[~ok].groupby("pack_id").size().sort_values(ascending=False).head(10).to_string())

def pooled(d, label):
    q = d.A_heating_c.quantile([.25, .5, .75])
    print(f"{label}: median A={q[.5]:.2f} IQR=[{q[.25]:.2f},{q[.75]:.2f}] n={len(d)}")
pooled(f, "all packs      ")
pooled(f[f.tau >= 400], "excl. tau<400 (b54)")
pooled(f.groupby("pack_id").head(30), "first 30 blocks/pack (early life)")
pooled(f.groupby("pack_id").tail(30), "last 30 blocks/pack (late life)")
print("pred I^2R0/h = 5.48 C (R0=0.04138, h=0.06798, I=3 A)")

rows = []
for p, g in f.groupby("pack_id"):
    n = len(g); k = max(5, n // 10)
    sl = np.polyfit(g.block_idx, g.A_heating_c, 1)[0] * 100 if n > 10 else np.nan
    rows.append(dict(pack_id=p, n=n, tau=g.tau.iloc[0], A_med=g.A_heating_c.median(),
                     A_early=g.A_heating_c.head(k).median(), A_late=g.A_heating_c.tail(k).median(),
                     slope_per100=sl))
by = pd.DataFrame(rows); by["A_growth_ratio"] = by.A_late / by.A_early
by.to_csv(f"{R}/charge_A_by_pack.csv", index=False)
print("\nA growth late/early (R0 growth proxy):\n", by[["pack_id","n","A_early","A_late","A_growth_ratio"]].round(2).to_string(index=False))
print("median growth ratio:", round(by.A_growth_ratio.median(), 2))

# capacity proxy: voltage-limited, non-capped blocks only (duration ~ Ah / 3 A)
c = f[(f.vc_max >= 8.625) & ~f.time_capped].copy()
cap = []
for p, g in c.groupby("pack_id"):
    ref = g.duration_s.head(10).median()
    g = g.assign(cap_rel=g.duration_s / ref, ah_est=g.duration_s * 3.0 / 3600)
    cap.append(g[["pack_id","block_idx","t0","duration_s","ah_est","cap_rel"]])
cap = pd.concat(cap); cap.to_csv(f"{R}/charge_capacity_proxy.csv", index=False)
last = cap.groupby("pack_id").tail(5).groupby("pack_id").cap_rel.median()
print("\ncharge-duration retention over last 5 CV-limited blocks (cap_rel):\n", last.round(3).to_string())
f.to_csv(f"{R}/charge_block_filtered.csv", index=False)
