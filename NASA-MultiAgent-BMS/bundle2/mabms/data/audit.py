"""Schema-agnostic dataset audit. Makes NO assumptions about column names."""
from __future__ import annotations
import json, zipfile
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

TABULAR = {".csv", ".txt", ".tsv"}
OTHER = {".mat", ".h5", ".hdf5", ".json", ".xlsx", ".xls", ".pdf", ".md", ".docx"}

KEYWORDS = {  # which variables could support which task (keyword match on column names, to be reviewed by a human)
    "time": ["time"], "voltage": ["volt"], "current": ["curr"], "temperature": ["temp"],
    "mode": ["mode"], "mission": ["mission", "type"], "soc": ["soc", "charge_state"],
    "capacity": ["capac"], "soh": ["soh", "health"], "cooling": ["cool", "fan"],
}


def extract_if_zip(path: str | Path, out_dir: str | Path) -> Path:
    path, out_dir = Path(path), Path(out_dir)
    if path.suffix.lower() == ".zip":
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(out_dir)
        return out_dir
    return path


def discover_files(root: str | Path) -> List[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))


def audit_table(path: Path, nrows: int | None = None) -> Dict[str, Any]:
    sep = "\t" if path.suffix.lower() == ".tsv" else None
    df = pd.read_csv(path, sep=sep, engine="python", nrows=nrows)
    info: Dict[str, Any] = {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": {}}
    for c in df.columns:
        s = df[c]
        d: Dict[str, Any] = {"dtype": str(s.dtype), "nan_frac": float(s.isna().mean())}
        if pd.api.types.is_numeric_dtype(s):
            d.update(min=float(s.min()), max=float(s.max()), mean=float(s.mean()),
                     n_unique=int(s.nunique()))
            if s.nunique() <= 10:
                d["unique_values"] = [float(v) for v in s.dropna().unique()[:10]]
        info["columns"][str(c)] = d
    tcols = [c for c in df.columns if "time" in str(c).lower() and pd.api.types.is_numeric_dtype(df[c])]
    if tcols:
        t = df[tcols[0]].to_numpy(float)
        dt = np.diff(t)
        info["time_col_guess"] = str(tcols[0])
        info["time_monotonic"] = bool(np.all(dt >= 0))
        if len(dt):
            info["dt_stats"] = {"median": float(np.median(dt)), "min": float(dt.min()),
                                "max": float(dt.max()), "p99": float(np.percentile(dt, 99))}
    ccols = [c for c in df.columns if "curr" in str(c).lower() and pd.api.types.is_numeric_dtype(df[c])]
    if ccols:
        i = df[ccols[0]].to_numpy(float)
        info["current_sign_fraction"] = {"positive": float((i > 0.05).mean()),
                                         "negative": float((i < -0.05).mean()),
                                         "near_zero": float((np.abs(i) <= 0.05).mean())}
    return info


def audit_other(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"note": "non-tabular; inspect manually"}
    if path.suffix.lower() == ".mat":
        try:
            from scipy.io import loadmat
            m = loadmat(path, squeeze_me=True, struct_as_record=False)
            info["mat_keys"] = [k for k in m if not k.startswith("__")]
        except NotImplementedError:
            import h5py
            with h5py.File(path, "r") as f:
                info["h5_top_keys"] = list(f.keys())
        except Exception as e:  # noqa
            info["error"] = repr(e)
    return info


def audit_directory(root: str | Path, nrows: int | None = None) -> Dict[str, Any]:
    root = Path(root)
    files = discover_files(root)
    inv: Dict[str, Any] = {"root": str(root), "n_files": len(files), "files": {}}
    for p in files:
        rel = str(p.relative_to(root))
        rec: Dict[str, Any] = {"size_bytes": p.stat().st_size, "suffix": p.suffix.lower(),
                               "parent": str(p.parent.relative_to(root))}
        try:
            if p.suffix.lower() in TABULAR:
                rec.update(audit_table(p, nrows))
            elif p.suffix.lower() in OTHER:
                rec.update(audit_other(p))
        except Exception as e:  # noqa
            rec["error"] = repr(e)
        inv["files"][rel] = rec
    # variable-support table
    cols = set()
    for r in inv["files"].values():
        cols.update(r.get("columns", {}).keys())
    inv["all_columns"] = sorted(cols)
    inv["variable_support"] = {k: [c for c in cols if any(w in c.lower() for w in ws)]
                               for k, ws in KEYWORDS.items()}
    inv["files_by_parent"] = pd.Series([r["parent"] for r in inv["files"].values()]).value_counts().to_dict()
    return inv


def save_inventory(inv: Dict[str, Any], manifest_dir: str | Path) -> None:
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "inventory.json").write_text(json.dumps(inv, indent=2))
    rows = []
    for rel, r in inv["files"].items():
        rows.append({"file": rel, "parent": r.get("parent"), "suffix": r.get("suffix"),
                     "size_bytes": r.get("size_bytes"), "n_rows": r.get("n_rows"),
                     "n_cols": r.get("n_cols"), "columns": ";".join(r.get("columns", {}).keys()),
                     "dt_median": r.get("dt_stats", {}).get("median"),
                     "dt_max": r.get("dt_stats", {}).get("max"),
                     "time_monotonic": r.get("time_monotonic"), "error": r.get("error")})
    pd.DataFrame(rows).to_csv(manifest_dir / "inventory.csv", index=False)
