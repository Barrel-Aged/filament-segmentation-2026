import hashlib
import json
import os
import random
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from . import ROOT, HOST_IDENTITY

# Set before initializing any CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
import numpy as np
import torch

from filament.common import sha256


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def gpu_inventory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,driver_version",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=20)
    rows = []
    for line in result.stdout.strip().splitlines():
        index, name, total, free, utilization, driver = [s.strip() for s in line.split(",")]
        rows.append({"index": int(index), "name": name, "total_mib": int(total), "free_mib": int(free),
                     "utilization_percent": int(utilization), "driver": driver})
    return rows


def configure_gpu(config):
    inventory = gpu_inventory()
    if len(inventory) != 1 or inventory[0]["name"] != "NVIDIA GeForce RTX 4070 SUPER":
        raise RuntimeError("GPU inventory differs from the configured training host")
    gpu = inventory[0]
    if gpu["free_mib"] < config["minimum_free_gpu_mib"] or gpu["utilization_percent"] > config["maximum_initial_gpu_utilization"]:
        raise RuntimeError("GPU is busy or has insufficient free memory; leave other jobs untouched")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU or local-GPU fallback")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["gpu_memory_fraction"], 0)
    seed_everything(config["seed"])
    # Explicit runtime evidence beyond nvidia-smi: CUDA matrix operation and sync.
    probe = torch.ones((32, 32), device="cuda")
    if float((probe @ probe).mean().item()) != 32:
        raise RuntimeError("CUDA probe failed")
    torch.cuda.synchronize()
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                                check=True, capture_output=True, text=True).stdout.strip()
    return {"host": HOST_IDENTITY, "gpu": gpu, "cuda_probe_passed": True, "torch": torch.__version__,
            "torchvision": version("torchvision"), "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "process_inventory": processes,
            "memory_fraction_cap": config["gpu_memory_fraction"], "utc": utc_now()}


def source_contract(config, weights, fixture_hash=None):
    files = {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "src").rglob("*.py"))}
    contract = {"config": config, "source_files": files, "weights_sha256": weights["sha256"],
                "training_lock_sha256": sha256(ROOT / "training/uv.lock"),
                "split_sha256": sha256(ROOT / "manifests/split-v2.json"),
                "dataset_sha256": sha256(ROOT / "manifests/dataset-v1.json"),
                "fixture_sha256": fixture_hash}
    contract["fingerprint"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return contract


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def state_digest(value):
    digest = hashlib.sha256()
    def visit(obj):
        if isinstance(obj, torch.Tensor):
            array = obj.detach().cpu().contiguous()
            digest.update(str((array.dtype, tuple(array.shape))).encode())
            digest.update(array.numpy().tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                digest.update(str(key).encode())
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
        else:
            digest.update(repr(obj).encode())
    visit(value)
    return digest.hexdigest()
