"""Config loading and reproducibility helpers. Set env MABMS_CONFIG=configs/calibrated.yaml to switch every script to a calibrated config."""
from __future__ import annotations
import copy, os, random
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml


def load_config(path: str | Path = "configs/default.yaml") -> Dict[str, Any]:
    if str(path) == "configs/default.yaml":
        path = os.environ.get("MABMS_CONFIG", path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def deep_copy(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(cfg)
