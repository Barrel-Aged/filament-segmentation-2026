"""Build candidate tables from a map file (n,2,2048,2048) uint8 + its .json index. CPU only.

usage: build_cands.py <maps.npy> <out_dir> [--test]
"""
import json, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfil.cands import candidates, gt_label_maps, region_of

PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RAW = os.path.join(PARENT, 'data', 'raw', 'MAGFiLO_1.0_Kaggle_2026')
ANN = os.path.join(PARENT, 'data', 'annotations', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
SPLIT = os.path.join(PARENT, 'claude', 'configs', 'split-v2.json')

_G = {}


def _init(maps_path, test):
    _G['maps'] = np.load(maps_path, mmap_mode='r')
    _G['test'] = test
    if not test:
        ann = json.load(open(ANN))
        by = {}
        for a in ann['annotations']:
            by.setdefault(a['image_id'], []).append(a)
        _G['by'] = by
        _G['recs'] = {o['file_name']: o['annotator_image_ids'] for o in json.load(open(SPLIT))['observations']}


def disk_stats(img):
    thr = 0.3 * np.percentile(img, 99)
    v = img[img > thr].astype(np.float32)
    return float(v.mean()), float(v.std() + 1e-6)


def _one(args):
    j, fname, out_dir = args
    dst = os.path.join(out_dir, fname.replace('.jpeg', '.npz'))
    if os.path.exists(dst):
        return fname
    m = np.asarray(_G['maps'][j])
    sub = 'test/test_images' if _G['test'] else 'train/train_images'
    img = cv2.imread(os.path.join(RAW, sub, fname), cv2.IMREAD_GRAYSCALE)
    dm, ds = disk_stats(img)
    gts = None
    ngt = np.zeros(0, np.int64)
    if not _G['test']:
        recs = _G['recs'][fname]
        gts = gt_label_maps(_G['by'], recs)
        ngt = np.asarray([len(a) - 1 for _, a in gts])
    aux = [] if gts is not None else None
    F, par, Hh, keys = candidates(m[0], m[1], img, dm, ds, gts, aux)
    extra = {'aux': np.asarray(aux, np.float32).reshape(len(F), len(gts), 3)} if gts is not None else {}
    np.savez_compressed(dst, F=F, parents=par, reg=region_of(m[0], keys), **extra, hits=Hh if Hh is not None else np.zeros((len(F), 0), np.float32), ngt=ngt)
    return fname


def main():
    maps_path, out_dir = sys.argv[1], sys.argv[2]
    test = '--test' in sys.argv
    idx = json.load(open(maps_path + '.json'))
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(j, f, out_dir) for j, f in enumerate(idx['files']) if not os.path.exists(os.path.join(out_dir, f.replace('.jpeg', '.npz')))]
    if not jobs:
        print('nothing to do'); return
    with ProcessPoolExecutor(int(os.environ.get('NPROC', 14)), initializer=_init, initargs=(maps_path, test)) as ex:
        for n, f in enumerate(ex.map(_one, jobs, chunksize=2)):
            if n % 20 == 0:
                print(n, f, flush=True)
    print('done', len(jobs))


if __name__ == '__main__':
    main()
