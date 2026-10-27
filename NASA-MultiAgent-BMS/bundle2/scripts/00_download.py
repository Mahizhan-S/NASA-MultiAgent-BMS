"""Download the official archive (URL from NASA catalog metadata) and extract to data/raw."""
import sys, urllib.request, zipfile
from pathlib import Path
from mabms.config import load_config

cfg = load_config(); raw = Path(cfg["data"]["raw_dir"]); raw.mkdir(parents=True, exist_ok=True)
ext = raw / "extracted"
alt_src = Path("/Users/mahizhan/Documents/Final_project/battery_alt_dataset")
zp = raw / "battery_alt_dataset.zip"

if not any(ext.rglob("*.csv")):
    if alt_src.exists():
        print(f"Using local dataset from {alt_src}")
        for p in alt_src.rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                rel = p.relative_to(alt_src)
                target = ext / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    import os
                    try:
                        os.link(p, target)
                    except OSError:
                        import shutil
                        shutil.copy2(p, target)
        print("Populated", ext, "from local source.")
    else:
        if not zp.exists():
            print("Downloading", cfg["data"]["download_url"]); urllib.request.urlretrieve(cfg["data"]["download_url"], zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(ext)
        print("Extracted to", ext)
else:
    print("Dataset already present in", ext)

