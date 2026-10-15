"""Dataset-based predictive maintenance demonstration."""
import os
from pathlib import Path

# Keep plotting configuration inside the project, including on restricted PCs.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
