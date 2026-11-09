"""Fail closed before GPU frameworks or commands can be launched."""
import getpass
import platform
import socket
import subprocess

from .common import read_json


def require_gpu_host(config_path):
    config = read_json(config_path)
    actual = (platform.system(), socket.gethostname().split(".")[0].lower(), getpass.getuser())
    expected = (config["remote_os"], config["remote_hostname"].lower(), config["remote_user"])
    if config.get("local_gpu_allowed") is not False or actual != expected:
        raise RuntimeError(f"GPU work is permitted only on {config['ssh_target']}; current host identity is {actual}.")
    return {"os": actual[0], "hostname": actual[1], "user": actual[2]}


def remote_preflight(config_path):
    identity = require_gpu_host(config_path)
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    return {"identity": identity, "gpu_inventory": result.stdout.strip(), "training_started": False}
