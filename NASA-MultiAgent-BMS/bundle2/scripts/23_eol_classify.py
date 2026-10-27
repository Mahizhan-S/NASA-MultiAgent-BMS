"""v7.1 part 2: conservative EOL labelling. Inputs: end_of_data_checks.csv (+ optional charge_capacity_proxy.csv)."""
import pandas as pd, sys
R = sys.argv[1] if len(sys.argv) > 1 else "results"
e = pd.read_csv(f"{R}/end_of_data_checks.csv")
def lab(r):
    if r.last_rel_dur < 0.2 or r.v_first_last < 2.0: return "event_strong"     # collapsed / pack open at discharge start
    if r.last_rel_dur < 0.7: return "ambiguous"                                  # could be a high-current draw
    if r.min_rel_dur_last6 < 0.2: return "transient_collapse"                    # a collapse earlier in last 6 discharges
    return "no_failure_evidence"
e["label"] = e.apply(lab, axis=1)
try:
    cap = pd.read_csv(f"{R}/charge_capacity_proxy.csv")
    e = e.merge(cap.groupby("pack_id").tail(5).groupby("pack_id").cap_rel.median().rename("charge_cap_ret_last5"),
                on="pack_id", how="left")
    e["near_80pct"] = e.charge_cap_ret_last5.between(0.72, 0.88)
except FileNotFoundError:
    pass
e.loc[e.pack_id == "battery23", "label"] = "event_charger_dead"   # last charge block vc_max=-0.111 V (verified manually from block table)
e.to_csv(f"{R}/eol_labels_v71.csv", index=False)
print(e[["pack_id","last_rel_dur","min_rel_dur_last6","v_first_last","label"]
        + [c for c in ["charge_cap_ret_last5","near_80pct"] if c in e]].round(3).sort_values("label").to_string(index=False))
print(e.label.value_counts().to_string())
