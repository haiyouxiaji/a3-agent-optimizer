from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SIM_SRC = ROOT.parent / "a3_dual_arm_sim" / "src"
if SIM_SRC.is_dir() and str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))
