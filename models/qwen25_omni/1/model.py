from pathlib import Path
import sys

_BACKENDS = Path(__file__).resolve().parents[3] / "backends"
if str(_BACKENDS) not in sys.path:
    sys.path.insert(0, str(_BACKENDS))

from qwen_omni_backend import TritonPythonModel as _Backend
from telemetry import instrument

TritonPythonModel = instrument(_Backend)
