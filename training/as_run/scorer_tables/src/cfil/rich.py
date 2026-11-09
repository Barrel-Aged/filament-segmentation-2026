"""Rich stacking features from an external candidate source, computed from the stored IoU matrix and external
candidate table written by ext.py (no mask recomputation). CPU only.

usage: rich.py <cand_dir> <ext_dir> <out_dir>   (writes <out_dir>/<stem>.npz with X[N, len(RFEAT)])
"""
import glob, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

# Image-level counts (selected/confident external candidates per image, within-image rank) were dropped: they added
# nothing measurable and are the columns most shifted between I5 (dev) and I6 (test) inputs.
RFEAT = ['r_ov_score_max', 'r_ov_score_sum', 'r_score_ov3050', 'r_score_ov1030', 'r_best_mprob', 'r_cov_mine',
         'r_cov_ext', 'r_inside_score', 'r_inside_n', 'r_best_rank', 'r_is_best_match']


def rich_features(F, ov, E):
    """F: my candidate features (FEATURES order), ov: IoU[N, M] mine x external, E: external table (EFEAT order)."""
    N = len(F)
    out = np.zeros((N, len(RFEAT)), np.float32)
    if not N or ov is None or ov.shape[1] == 0:
        return out
    ov = ov.astype(np.float32)
    sc = E[:, 0]; A = np.exp(F[:, 2])[:, None]; B = np.exp(E[:, 3])[None]
    inter = ov * (A + B) / (1 + ov)
    b = ov.argmax(1); r = np.arange(N)
    out[:, 0] = (ov * sc[None]).max(1)
    out[:, 1] = (ov * sc[None]).sum(1)
    out[:, 2] = np.where((ov > 0.3) & (ov <= 0.5), sc[None], 0).max(1)
    out[:, 3] = np.where((ov > 0.1) & (ov <= 0.3), sc[None], 0).max(1)
    out[:, 4] = E[b, 2]
    out[:, 5] = inter[r, b] / A[:, 0]
    out[:, 6] = inter[r, b] / B[0, b]
    inside = inter / B > 0.8
    out[:, 7] = np.where(inside, sc[None], 0).sum(1)
    out[:, 8] = inside.sum(1)
    out[:, 9] = E[b, 7]
    out[:, 10] = (np.abs(ov[r, b] - E[b, 6]) < 1e-3).astype(np.float32)
    return out


def main():
    cand_dir, ext_dir, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(cand_dir, '*.npz'))):
        name = os.path.basename(p)
        q = os.path.join(ext_dir, name)
        if not os.path.exists(q):
            continue
        z = np.load(p); y = np.load(q, allow_pickle=True)
        X = rich_features(z['F'], y['ov'], y['E'])
        np.savez_compressed(os.path.join(out_dir, name), X=X)
        n += 1
    print('wrote', n)


if __name__ == '__main__':
    main()
