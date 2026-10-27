"""v7.3 part 1: cohorts, absolute SOH, endpoint table (one row per pack).
Inputs : data/processed/cycle_table.csv, results/eol_labels_v71.csv
Output : results/pack_endpoints_v73.csv
Usage  : python scripts/24_pack_endpoints.py [--basis abs|rel] [--thr 0.8] [--gap-ah 60]
"""
import argparse, numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--ct", default="data/processed/cycle_table.csv")
ap.add_argument("--labels", default="results/eol_labels_v71.csv")
ap.add_argument("--out", default="results/pack_endpoints_v73.csv")
ap.add_argument("--basis", default="abs", choices=["abs", "rel"])
ap.add_argument("--thr", type=float, default=0.80)
ap.add_argument("--q0", type=float, default=2.45)
ap.add_argument("--gap-ah", type=float, default=60.0, help="max throughput between last ref check and true failure (~20 cycles)")
a = ap.parse_args()

ct = pd.read_csv(a.ct).sort_values(["pack_id", "ref_idx"])
lab = pd.read_csv(a.labels).set_index("pack_id").label
ct["soh_abs"] = ct.capacity_ah / a.q0
ct["soh_rel"] = ct.soh
ct["soh_use"] = ct["soh_abs" if a.basis == "abs" else "soh_rel"]
FAIL = {"event_strong", "event_probable", "event_charger_dead"}
rows = []
for p, g in ct.groupby("pack_id"):
    g = g.reset_index(drop=True)
    cohort = "recommissioned" if g.cap_ref_ah.median() < 2.0 else "original"
    below = (g.soh_use <= a.thr).to_numpy()
    cross_i = None
    for i in range(len(g)):
        if below[i] and (i == len(g) - 1 or below[i + 1]):   # confirmed by next ref, or last ref
            cross_i = i; break
    confirmed = cross_i is not None and cross_i < len(g) - 1
    # noisy: SOH rises >0.03 after a drop
    d = np.diff(g.soh_use.to_numpy()); noisy = bool((d > 0.03).any())
    trunc = bool("v_end" in g and g.v_end.iloc[-1] > 2.7)  # last segment ended well above cutoff
    fail = lab.get(p, "") in FAIL
    last_thr = float(g.throughput_ah.iloc[-1])
    cross_thr = float(g.throughput_ah.iloc[cross_i]) if cross_i is not None else np.nan
    rows.append(dict(pack_id=p, cohort=cohort, n_ref=len(g), label=lab.get(p, "?"),
                     last_thr=last_thr, last_soh_abs=g.soh_abs.iloc[-1], last_soh_rel=g.soh_rel.iloc[-1],
                     cross_thr=cross_thr, cross_confirmed=confirmed, failure=fail,
                     fail_thr_lo=last_thr if fail else np.nan, fail_thr_hi=last_thr + a.gap_ah if fail else np.nan,
                     B_time=min(x for x in [cross_thr, last_thr] if not np.isnan(x)),
                     B_event=int(fail or cross_i is not None),
                     low_trajectory=len(g) <= 7, noisy_soh=noisy, truncated_last=trunc,
                     fail_before_threshold=bool(fail and (cross_i is None or cross_i == len(g) - 1))))
e = pd.DataFrame(rows)
e["use_in_main"] = (e.cohort == "original")
e["use_in_regression"] = e.use_in_main & ~e.low_trajectory & e.cross_thr.notna()
e.to_csv(a.out, index=False)
print(f"basis={a.basis} thr={a.thr}")
print(e.groupby("cohort").agg(n=("pack_id", "size"), crossers=("cross_thr", lambda s: s.notna().sum()),
                              failures=("failure", "sum")).to_string())
print("failures that happened BEFORE SOH threshold:", int(e.fail_before_threshold.sum()), "of", int(e.failure.sum()))
print("flags -> low_trajectory:", e[e.low_trajectory].pack_id.tolist(),
      "| noisy:", e[e.noisy_soh].pack_id.tolist(), "| truncated_last:", e[e.truncated_last].pack_id.tolist())
print("regression set:", e[e.use_in_regression].pack_id.tolist())
