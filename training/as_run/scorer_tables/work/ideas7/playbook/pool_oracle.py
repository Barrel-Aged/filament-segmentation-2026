"""Playbook diagnostic 4 (oracle only): headroom of pooling the component trees of other dense models (cnxL, R34 OOF
maps) with the cnx tree under exact selection (stacked candidate pooling / instance-ensemble analog).
Per image: cnx tree nodes (base) + cnxL and R34 tree nodes that are not near-duplicates (IoU < 0.9) of any earlier node;
per-record hits as in cands.candidates; conflicts = pixel overlap. Writes work/ideas7/playbook/xpool/<stem>.npz in the
xcands layout (F = 28 tree features of each node computed on its own map; fam 0 cnx, 5 cnxL, 6 R34).
Dev folds 0-3 only, CPU only. usage: pool_oracle.py [--nproc=5]
"""
import json, os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = 'D:/FilamentSegmentationKaggle/claude'
sys.path.insert(0, ROOT + '/src'); sys.path.insert(0, ROOT + '/work/ideas7/playbook')
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from cfil.cands import LEVELS, candidates, gt_label_maps
from cfil.build_cands import disk_stats, RAW
import xcands as X
import xcands2 as X2

OUT = ROOT + '/work/ideas7/playbook/xpool'
W = 2048
_G = {}


def _init():
    X._init()
    _G['other'] = {}
    for name in ('cnxL', 'r34'):
        for k in range(4):
            mp = ROOT + f'/work/maps/{name}-f{k}/oof.npy'
            files = json.load(open(mp + '.json'))['files']
            arr = np.load(mp, mmap_mode='r')
            for j, f in enumerate(files):
                _G['other'][(name, f)] = (arr, j)


def tree_sets(P, keys):
    labs = {}
    out = []
    for (li, kk, sl) in keys:
        if li not in labs:
            labs[li] = cv2.connectedComponents((P > X.THR[li]).astype(np.uint8), connectivity=8)[1]
        ys, xs = np.nonzero(labs[li][sl] == kk)
        out.append(((ys + sl[0].start) * W + xs + sl[1].start).astype(np.int64))
    return out


def one(args):
    k, j, fname = args
    stem = fname[:-5]
    dst = os.path.join(OUT, stem + '.npz')
    if os.path.exists(dst):
        return fname, None
    m = np.asarray(X._G['maps'][k][j])
    img = cv2.imread(os.path.join(RAW, 'train/train_images', fname), cv2.IMREAD_GRAYSCALE)
    dm, ds = disk_stats(img)
    gts = gt_label_maps(X._G['by'], X._G['recs'][fname])
    F, par, Hh, keys = candidates(m[0], m[1], img, dm, ds, gts)
    sets = tree_sets(m[0], keys)
    Fs, pars, Hs, fams = [F], [par], [Hh], [np.zeros(len(F), np.int8)]
    for code, name in ((5, 'cnxL'), (6, 'r34')):
        arr, jj = _G['other'][(name, fname)]
        m2 = np.asarray(arr[jj])
        F2, par2, H2, keys2 = candidates(m2[0], m2[1], img, dm, ds, gts)
        s2 = tree_sets(m2[0], keys2)
        if len(s2) and len(sets):
            Ca, la = X2.csr_of(s2); Cb, lb = X2.csr_of(sets)
            keep = X2.iou_sparse(Ca, la, Cb, lb).max(1) < 0.9
        else:
            keep = np.ones(len(s2), bool)
        idx = np.flatnonzero(keep)
        Fs.append(F2[idx]); Hs.append(H2[idx]); pars.append(np.full(len(idx), -1)); fams.append(np.full(len(idx), code, np.int8))
        sets += [s2[i] for i in idx]
    nb = len(F)
    Fall = np.concatenate(Fs); Hall = np.concatenate(Hs); fam = np.concatenate(fams)
    parents = np.concatenate(pars)
    C, l = X2.csr_of(sets)
    ov = (C @ C.T).tocoo()
    u = ov.row < ov.col
    conf = np.stack([ov.row[u], ov.col[u]], 1).astype(np.int32)
    np.savez_compressed(dst, F=Fall, fam=fam, parents=parents, hits=Hall, ngt=np.asarray([len(a) - 1 for _, a in gts]),
                        conf=conf, nb=nb)
    return fname, (nb, int((fam == 5).sum()), int((fam == 6).sum()))


def main():
    opt = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for k in range(4):
        mp = ROOT + f'/work/maps/cnx-f{k}/oof.npy'
        jobs += [(k, j, f) for j, f in enumerate(json.load(open(mp + '.json'))['files'])]
    t0 = time.time()
    with ProcessPoolExecutor(int(opt.get('nproc', 5)), initializer=_init) as ex:
        for n, (f, info) in enumerate(ex.map(one, jobs, chunksize=1)):
            if n % 50 == 0:
                print(n, f, info, f'elapsed {time.time() - t0:.0f}s', flush=True)
    print('done', len(jobs), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
