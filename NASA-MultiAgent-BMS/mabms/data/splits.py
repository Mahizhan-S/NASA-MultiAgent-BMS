"""Pack-level splits only (never split windows of the same pack across sets)."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
from sklearn.model_selection import GroupKFold


def split_packs(pack_ids: List[str], test_fraction: float, val_fraction: float, seed: int) -> Dict[str, List[str]]:
    ids = sorted(set(pack_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * test_fraction))) if n >= 3 else 0
    n_val = max(1, int(round(n * val_fraction))) if n >= 5 else 0
    return {"test": sorted(ids[:n_test]), "val": sorted(ids[n_test:n_test + n_val]),
            "train": sorted(ids[n_test + n_val:])}


def grouped_folds(groups, n_splits: int):
    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))
