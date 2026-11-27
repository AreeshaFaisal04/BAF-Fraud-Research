# saves the library versions and the machine info
# python scripts/00_environment.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import platform

import matplotlib
import numpy
import pandas
import scipy
import sklearn

from src.config import TAB_DIR

TAB_DIR.mkdir(parents=True, exist_ok=True)
cpu = platform.processor() or "unknown"
try:
    with open("/proc/cpuinfo") as f:
        names = [l.split(":", 1)[1].strip() for l in f if l.startswith("model name")]
    if names:
        cpu = names[0]
except OSError:
    pass
try:
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
except (ValueError, OSError, AttributeError):
    ram_gb = None

env = {
    "python": platform.python_version(),
    "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
    "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
    "os": platform.platform(), "cpu": cpu, "cpu_count": os.cpu_count(),
    "ram_gb": round(ram_gb, 1) if ram_gb else None,
    "colab": "google.colab" in sys.modules or os.path.exists("/content"),
}
with open(TAB_DIR / "environment.json", "w") as f:
    json.dump(env, f, indent=2)
print(json.dumps(env, indent=2))
