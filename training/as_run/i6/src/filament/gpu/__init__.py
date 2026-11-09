"""Remote-only training modules. Reject unauthorized hosts before torch imports."""
from pathlib import Path
from filament.compute import require_gpu_host

ROOT = Path(__file__).resolve().parents[3]
HOST_IDENTITY = require_gpu_host(ROOT / "configs/compute.json")
