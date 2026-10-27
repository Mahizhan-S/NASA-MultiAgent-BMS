"""Replay measured current of full-discharge segments through the surrogate; fit R0,R1,C1 (voltage) and C_th,h (temp).
Reports voltage/temperature RMSE per pack. Uses TRAIN packs for fitting, reports on VAL; TEST untouched."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from mabms.config import load_config
from mabms.data.loader import load_all
from mabms.data.segments import label_segments
from mabms.environment.battery_model import BatteryModel, ModelParams
from mabms.estimation.soc_estimator import OCVCurve

cfg = load_config(); fc = cfg["data"]["full_discharge"]; sc = cfg["data"]["segmentation"]
split = json.loads(Path(cfg["data"]["manifest_dir"], "pack_split.json").read_text())
packs = load_all(cfg["data"]["raw_dir"], cfg)

def segments(df):
    d = label_segments(df, sc["i_threshold_a"], sc["max_gap_s"]); out = []
    for _, g in d.groupby("seg_id"):
        if g.seg_code.iloc[0] == -1 and g.time_s.iloc[-1] - g.time_s.iloc[0] >= fc["min_duration_s"] \
           and g.voltage_v.iloc[0] >= fc["v_start_min"] and g.voltage_v.iloc[-1] <= fc["v_end_max"]:
            out.append(g)
    return out

def simulate(p, g, soc0):
    m = BatteryModel(p, soc0=soc0, temp0_c=g.temp_c.iloc[0], t_amb_c=g.temp_c.iloc[0])
    t, I = g.time_s.to_numpy(), g.current_a.to_numpy(); V, T = [], []
    for k in range(len(g)):
        dt = t[k] - t[k - 1] if k else 1.0
        o = m.step(I[k], 0.0, max(dt, 1e-3)); V.append(o["voltage"]); T.append(m.temp_c)
    return np.array(V), np.array(T)

base = ModelParams.from_config(cfg)
def make(x): 
    p = ModelParams(**base.__dict__); p.r0_ohm, p.r1_ohm, p.c1_f, p.c_th, p.h_w_per_k = x; return p
def residual(x, segs):
    p = make(x); r = []
    for g in segs:
        v, T = simulate(p, g, soc0=1.0)           # ASSUMPTION: full discharge starts at SOC=1 of its own capacity
        r += [v - g.voltage_v.to_numpy(), 0.1 * (T - g.temp_c.to_numpy())]
    return np.concatenate(r)
train_segs = [g for pid in split["train"] if pid in packs for g in segments(packs[pid])[:3]]
x0 = [base.r0_ohm, base.r1_ohm, base.c1_f, base.c_th, base.h_w_per_k]
sol = least_squares(residual, x0, args=(train_segs,), bounds=([1e-3, 1e-3, 100, 5, 1e-3], [1, 1, 1e5, 500, 1]))
print("fitted [r0,r1,c1,c_th,h]:", sol.x)
rows = []
for pid in split["val"] + split["test"][:0]:       # test packs deliberately excluded
    for g in segments(packs[pid])[:3]:
        v, T = simulate(make(sol.x), g, 1.0)
        rows.append({"pack_id": pid, "v_rmse": float(np.sqrt(np.mean((v - g.voltage_v.to_numpy()) ** 2))),
                     "t_rmse": float(np.sqrt(np.mean((T - g.temp_c.to_numpy()) ** 2)))})
df = pd.DataFrame(rows); df.to_csv("results/sim_validation_val.csv", index=False); print(df.groupby("pack_id").mean())
Path("results/sim_fitted_params.json").write_text(json.dumps(dict(zip(["r0_ohm", "r1_ohm", "c1_f", "c_th", "h_w_per_k"], map(float, sol.x))), indent=2))
