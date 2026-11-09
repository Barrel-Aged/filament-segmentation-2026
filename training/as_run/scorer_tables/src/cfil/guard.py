"""Fail closed unless running on the approved remote GPU host. Import before torch.
CFIL_LOCAL_GPU=1 is the one exception: the user approved training on this Windows PC's GPU overnight on 2026-10-29
while Codex uses the remote GPU. Every run that uses it says so on stdout."""
import getpass, json, os, platform, socket

_CFG = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'compute.json')


def require_gpu_host():
    c = json.load(open(_CFG))
    actual = (platform.system(), socket.gethostname().split('.')[0].lower(), getpass.getuser())
    if os.environ.get('CFIL_LOCAL_GPU') == '1':
        print(f'guard: CFIL_LOCAL_GPU=1 (user-approved local GPU) on {actual}', flush=True)
        return
    expected = (c['remote_os'], c['remote_hostname'].lower(), c['remote_user'])
    if c.get('local_gpu_allowed') is not False or actual != expected:
        raise RuntimeError(f'GPU work only on {c["ssh_target"]}; this host is {actual}')


require_gpu_host()
