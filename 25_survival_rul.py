"""v7.3 part 2: Kaplan-Meier (3 endpoint definitions) + leave-one-pack-out RUL on SOH-threshold crossers.
Inputs : results/pack_endpoints_v73.csv, data/processed/cycle_table.csv, optional data/manifests/pack_split.json
Outputs: results/km_v73.csv, results/rul_lopo_v73.csv, results/km_v73.png
"""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
E = pd.read_csv("results/pack_endpoints_v73.csv"); CT = pd.read_csv("data/processed/cycle_table.csv")
Q0, THR = 2.45, 0.80
M = E[E.use_in_main].copy()

def km(t, e):
    t = np.asarray(t, float); e = np.asarray(e, int); S = 1.0; out = [(0.0, 1.0)]
    for u in np.unique(t[e == 1]):
        S *= 1 - ((t == u) & (e == 1)).sum() / (t >= u).sum(); out.append((u, S))
    return pd.DataFrame(out, columns=["t", "S"])
defs = {"A: SOH<=0.8 only": (M.cross_thr.fillna(M.last_thr), M.cross_thr.notna().astype(int)),
        "B: first of SOH<=0.8 / failure": (M.B_time, M.B_event),
        "C: abrupt failure only": (M.last_thr, M.failure.astype(int))}
rows = []
for k, (t, e) in defs.items():
    K = km(t, e); K["endpoint"] = k; rows.append(K)
    med = K[K.S <= .5].t.min()
    print(f"{k}: events {int(np.sum(e))}/{len(M)} (main cohort), median throughput {med}")
    plt.step(K.t, K.S, where="post", label=k)
pd.concat(rows).to_csv("results/km_v73.csv", index=False)
plt.xlabel("Discharge throughput (Ah)"); plt.ylabel("Survival"); plt.grid(alpha=.3); plt.legend(fontsize=7)
plt.savefig("results/km_v73.png", dpi=150, bbox_inches="tight")

# ---- RUL regression on crossers (remaining throughput to threshold), leave-one-pack-out ----
reg = set(E[E.use_in_regression].pack_id)
d = CT[CT.pack_id.isin(reg)].sort_values(["pack_id", "ref_idx"]).copy()
d["soh_abs"] = d.capacity_ah / Q0
ct_thr = E.set_index("pack_id").cross_thr
d["rul"] = d.pack_id.map(ct_thr) - d.throughput_ah
d["fade_slope"] = d.groupby("pack_id").soh_abs.transform(lambda s: s.diff(3) / 1.0)  # per 3 refs
d["thr_diff3"] = d.groupby("pack_id").throughput_ah.transform(lambda s: s.diff(3))
d["fade_per100"] = d.fade_slope / d.thr_diff3 * 100
d = d[(d.rul > 0) & d.fade_per100.notna() & (d.soh_abs > THR)]
feats = ["throughput_ah", "soh_abs", "fade_per100"]
out = []
for p in sorted(d.pack_id.unique()):
    tr, te = d[d.pack_id != p], d[d.pack_id == p]
    med_cross = E[E.pack_id.isin(set(tr.pack_id))].cross_thr.median()
    base_pop = np.clip(med_cross - te.throughput_ah, 0, None)
    lin = np.clip((te.soh_abs - THR) / np.clip(-te.fade_per100 / 100, 1e-6, None), 0, 2000)
    mu, sd = tr[feats].mean(), tr[feats].std() + 1e-9
    rg = Ridge(alpha=1.0).fit((tr[feats] - mu) / sd, tr.rul)
    pr = np.clip(rg.predict((te[feats] - mu) / sd), 0, None)
    out.append(pd.DataFrame(dict(pack_id=p, thr=te.throughput_ah.values, rul=te.rul.values,
                                 pop=base_pop.values, linear=lin.values, ridge=pr)))
R = pd.concat(out); R.to_csv("results/rul_lopo_v73.csv", index=False)
print("\nLOPO RUL MAE (Ah) over", R.pack_id.nunique(), "packs,", len(R), "ref checks")
for m in ["pop", "linear", "ridge"]:
    print(f"  {m:7s} all={np.abs(R[m]-R.rul).mean():7.1f}   rul<=200: {np.abs(R[m]-R.rul)[R.rul<=200].mean():7.1f}")
print("  per-pack ridge MAE:", R.assign(e=(R.ridge - R.rul).abs()).groupby("pack_id").e.mean().round(0).to_dict())
sp = "data/manifests/pack_split.json"
if os.path.exists(sp):
    s = json.load(open(sp)); print("\nsplit keys:", list(s)[:5], "-> use for dev/test reporting (format not assumed)")
