"""Cross-model features: a second dense model's opinion about each of my tree candidates. CPU only.
For each image writes <out_dir>/<stem>.npz with X[N, len(XMFEAT)].

  xm_mean, xm_max, xm_frac50   second map's mask probability inside the candidate (mean, max, fraction > 0.5)
  xm_edge                      second map's mean probability on the candidate boundary
  xm_iou_same                  IoU with the second map's component at the same threshold level that overlaps it most
  xm_best_iou                  best IoU with any second-map component over all levels
usage: xmap.py <maps.npy> <second_maps.npy> <out_dir> [--test]   (both map files must list the same images)
"""
import json, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfil.cands import candidates, level_labels, LEVELS, H, W
from cfil.build_cands import disk_stats, RAW

XMFEAT = ['xm_mean', 'xm_max', 'xm_frac50', 'xm_edge', 'xm_iou_same', 'xm_best_iou']
PAD = 4
_K1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_G = {}


def _init(maps_path, second_path, test):
    _G['maps'] = np.load(maps_path, mmap_mode='r'); _G['second'] = np.load(second_path, mmap_mode='r'); _G['test'] = test
    files2 = json.load(open(second_path + '.json'))['files']
    _G['j2'] = {f: j for j, f in enumerate(files2)}


def _one(args):
    j, fname, out_dir = args
    dst = os.path.join(out_dir, fname.replace('.jpeg', '.npz'))
    if os.path.exists(dst):
        return fname
    m = np.asarray(_G['maps'][j]); Q = np.asarray(_G['second'][_G['j2'][fname]][0])
    img = cv2.imread(os.path.join(RAW, 'test/test_images' if _G['test'] else 'train/train_images', fname), cv2.IMREAD_GRAYSCALE)
    dm, ds = disk_stats(img)
    F, par, _, keys = candidates(m[0], m[1], img, dm, ds, None)
    mine, other = {}, {li: level_labels(Q, t)[1] for li, t in enumerate(LEVELS)}
    oarea = {li: np.bincount(lab.ravel()) for li, lab in other.items()}
    Qf = Q.astype(np.float32) / 255
    X = np.zeros((len(keys), len(XMFEAT)), np.float32)
    for i, (li, k, sl) in enumerate(keys):
        if li not in mine:
            mine[li] = level_labels(m[0], LEVELS[li])[1]
        y0, y1 = max(sl[0].start - PAD, 0), min(sl[0].stop + PAD, H)
        x0, x1 = max(sl[1].start - PAD, 0), min(sl[1].stop + PAD, W)
        c = mine[li][y0:y1, x0:x1] == k
        if not c.any():
            continue
        q = Qf[y0:y1, x0:x1]
        edge = c & ~cv2.erode(c.astype(np.uint8), _K1).astype(bool)
        ious = []
        for lj in range(len(LEVELS)):
            lab = other[lj][y0:y1, x0:x1]
            cnt = np.bincount(lab[c], minlength=1); cnt[0] = 0
            if cnt.max() == 0:
                ious.append(0.0); continue
            kk = int(cnt.argmax())
            ious.append(cnt[kk] / (c.sum() + oarea[lj][kk] - cnt[kk]))
        X[i] = [q[c].mean(), q[c].max(), (q[c] > 0.5).mean(), q[edge].mean() if edge.any() else 0, ious[li], max(ious)]
    np.savez_compressed(dst, X=X)
    return fname


def main():
    maps_path, second_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    test = '--test' in sys.argv
    idx = json.load(open(maps_path + '.json'))
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(j, f, out_dir) for j, f in enumerate(idx['files']) if not os.path.exists(os.path.join(out_dir, f.replace('.jpeg', '.npz')))]
    if not jobs:
        print('nothing to do'); return
    with ProcessPoolExecutor(int(os.environ.get('NPROC', 14)), initializer=_init, initargs=(maps_path, second_path, test)) as ex:
        for n, f in enumerate(ex.map(_one, jobs, chunksize=2)):
            if n % 40 == 0:
                print(n, f, flush=True)
    print('done', len(jobs))


if __name__ == '__main__':
    main()
