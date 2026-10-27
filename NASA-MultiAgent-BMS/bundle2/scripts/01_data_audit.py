"""Phase 1: audit. Usage: python scripts/01_data_audit.py [path_to_dir_or_zip]"""
import sys
from pathlib import Path
from mabms.config import load_config
from mabms.data.audit import audit_directory, extract_if_zip, save_inventory

cfg = load_config()
src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(cfg["data"]["raw_dir"])
root = extract_if_zip(src, Path(cfg["data"]["raw_dir"]) / "extracted")
inv = audit_directory(root)
save_inventory(inv, cfg["data"]["manifest_dir"])
print(f"files: {inv['n_files']}\nfiles by folder: {inv['files_by_parent']}")
print("\nall columns:", inv["all_columns"])
print("\nvariable support (keyword match, REVIEW MANUALLY):")
for k, v in inv["variable_support"].items(): print(f"  {k:12s} -> {v}")
n_packs_guess = len(inv["files"])
print(f"\nNumber of data files: {n_packs_guess} (compare with the documented 26 packs; one pack may span several files)")
print("Saved data/manifests/inventory.json and inventory.csv")
