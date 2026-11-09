"""External candidate source: Codex learned-ROI candidates (I5 OOF cells / I6 test), read-only. CPU only.

For each image writes <out_dir>/<stem>.npz with
  X[N_mine, len(XFEAT)]  stacking features for my tree candidates (overlap with external candidates)
  E[M, len(EFEAT)]       external candidate features, Eh[M, R] their per-record hit IoUs (train only)
  ov[N_mine, M]          IoU between my candidates and external ones (float16), for pooled selection

usage: ext.py <maps.npy> <out_dir> <cand_root>[,<cand_root>...] [--test] [--greedy-sel]
  cand_root: a directory with manifest.json (e.g. outputs/improvement-i5/f0-s20261014/candidates/outer)
  --greedy-sel: ignore the root's own selected set; use a disjoint greedy set (score >= 0.5, >= 70% of the mask unclaimed),
                so sources without a selector (I6 OOF) and their test counterparts get the same rule.
"""
import json, math, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from pycocotools import mask as mu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfil import cands as C
from cfil.cands import candidates, gt_label_maps, materialize_one
from cfil.build_cands import disk_stats, RAW, ANN, SPLIT

XFEAT = ['x_best_iou', 'x_best_score', 'x_best_det', 'x_score_ov50', 'x_n_ov30', 'x_cov_sel']
EFEAT = ['e_score', 'e_det', 'e_mprob', 'e_log_area', 'e_rho', 'e_thr', 'e_best_mine_iou', 'e_rank']
_sig = lambda z: 1 / (1 + math.exp(-z))
_G = {}


def load_roots(roots):
    by = {}
    for root in roots:
        for rec in json.load(open(os.path.join(root, 'manifest.json')))['records']:
            if rec.get('kind', 'clean') != 'clean':
                continue
            by[rec['file_name']] = os.path.join(root, rec['file'])
    return by


def greedy_sel(ext, score):
    taken = np.zeros((2048, 2048), bool); out = []
    for i in np.argsort(-score):
        if score[i] < 0.5:
            break
        m = mu.decode(ext[i]).astype(bool); free = m & ~taken
        if free.sum() >= 0.7 * m.sum() and free.sum() >= 20:
            taken |= free
            e = mu.encode(np.asfortranarray(free.astype(np.uint8)))
            out.append({'size': list(e['size']), 'counts': e['counts'].decode()})
    return out


def _init(maps_path, roots, test, greedy=False):
    _G['maps'] = np.load(maps_path, mmap_mode='r'); _G['test'] = test; _G['greedy'] = greedy
    _G['src'] = load_roots(roots)
    if not test:
        ann = json.load(open(ANN)); by = {}
        for a in ann['annotations']:
            by.setdefault(a['image_id'], []).append(a)
        _G['by'] = by
        _G['recs'] = {o['file_name']: o['annotator_image_ids'] for o in json.load(open(SPLIT))['observations']}


def _one(args):
    j, fname, out_dir = args
    dst = os.path.join(out_dir, fname.replace('.jpeg', '.npz'))
    if os.path.exists(dst):
        return fname
    r = json.load(open(_G['src'][fname]))
    (thr, rows), = r['views'].items()
    ext = [{'size': c['rle']['size'], 'counts': c['rle']['counts'].encode()} for c in rows]
    sel = greedy_sel(ext, np.asarray([c['score'] for c in rows], np.float32)) if _G['greedy'] else         r['predictions'].get('set_selector_pq', [])
    m = np.asarray(_G['maps'][j])
    sub = 'test/test_images' if _G['test'] else 'train/train_images'
    img = cv2.imread(os.path.join(RAW, sub, fname), cv2.IMREAD_GRAYSCALE)
    dm, ds = disk_stats(img)
    F, par, _, keys = candidates(m[0], m[1], img, dm, ds, None)
    P = m[0]; C._CC.clear()
    mine = [materialize_one(P, k) for k in keys]
    N, M = len(mine), len(ext)
    ov = np.asarray(mu.iou(mine, ext, [0] * M), np.float32).reshape(N, M) if N and M else np.zeros((N, M), np.float32)
    score = np.asarray([c['score'] for c in rows], np.float32)
    det = np.asarray([_sig(c['features'][1]) for c in rows], np.float32)
    selr = [{'size': s['size'], 'counts': s['counts'].encode()} for s in sel]
    X = np.zeros((N, len(XFEAT)), np.float32)
    if N and M:
        b = ov.argmax(1)
        X[:, 0] = ov.max(1); X[:, 1] = score[b]; X[:, 2] = det[b]
        X[:, 3] = np.where(ov > 0.5, score[None], 0).max(1)
        X[:, 4] = (ov > 0.3).sum(1)
    if N and selr:
        X[:, 5] = np.asarray(mu.iou(mine, selr, [0] * len(selr))).reshape(N, -1).max(1)
    E = np.zeros((M, len(EFEAT)), np.float32)
    order = np.argsort(-score); rk = np.empty(M); rk[order] = np.arange(M) / max(M - 1, 1)
    for i, c in enumerate(rows):
        E[i] = [c['score'], det[i], _sig(c['features'][2]), math.log(max(c['area'], 1)), c['rho'], float(thr),
                ov[:, i].max() if N else 0, rk[i]]
    Eh = np.zeros((M, 0), np.float32)
    if not _G['test']:
        gts = gt_label_maps(_G['by'], _G['recs'][fname])
        Eh = np.zeros((M, len(gts)), np.float32)
        for i, e in enumerate(ext):
            em = mu.decode(e).astype(bool)
            for ri, (lab, areas) in enumerate(gts):
                g = lab[em]; cnt = np.bincount(g, minlength=len(areas)); cnt[0] = 0
                k = int(cnt.argmax())
                if cnt[k]:
                    iou = cnt[k] / (em.sum() + areas[k] - cnt[k])
                    Eh[i, ri] = iou if iou > 0.5 else 0
    np.savez_compressed(dst, X=X, E=E, Eh=Eh, ov=ov.astype(np.float16), ext_rle=np.asarray([c['rle']['counts'] for c in rows], object),
                        ext_sel=np.asarray([s['counts'] for s in sel], object))
    return fname


def main():
    maps_path, out_dir, roots = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
    test = '--test' in sys.argv; greedy = '--greedy-sel' in sys.argv
    idx = json.load(open(maps_path + '.json'))
    have = load_roots(roots)
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(j, f, out_dir) for j, f in enumerate(idx['files']) if not os.path.exists(os.path.join(out_dir, f.replace('.jpeg', '.npz'))) if f in have]
    print('images', len(idx['files']), 'with external candidates', len(jobs), flush=True)
    if not jobs:
        print('nothing to do'); return
    with ProcessPoolExecutor(int(os.environ.get('NPROC', 14)), initializer=_init, initargs=(maps_path, roots, test, greedy)) as ex:
        for n, f in enumerate(ex.map(_one, jobs, chunksize=2)):
            if n % 20 == 0:
                print(n, f, flush=True)
    print('done', len(jobs))


if __name__ == '__main__':
    main()
