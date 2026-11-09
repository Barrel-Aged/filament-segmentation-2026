"""Build memory-mapped training/test arrays on the remote host (CPU only).

Outputs in <root>/cache:
  train_img.u8  (N,2048,2048) raw grayscale
  train_soft.u8 (N,2048,2048) annotator-mean filled mask * 255
  train_spine.u8 (N,2048,2048) annotator-mean spine line (width 3) * 255
  test_img.u8   (M,2048,2048)
  meta.json     file names, folds, groups, records, per-image disk stats
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from pycocotools import mask as mu

H = W = 2048


def disk_stats(img):
    thr = 0.3 * np.percentile(img, 99)
    m = img > thr
    v = img[m].astype(np.float32)
    return float(v.mean()), float(v.std() + 1e-6), float(thr)


def record_masks(anns):
    """Union mask and spine raster for one annotator record."""
    full = np.zeros((H, W), np.uint8)
    spine = np.zeros((H, W), np.uint8)
    for a in anns:
        segs = json.loads(a['segmentation']) if isinstance(a['segmentation'], str) else a['segmentation']
        rles = mu.frPyObjects([s for s in segs if len(s) >= 6], H, W)
        full |= mu.decode(mu.merge(rles)).astype(np.uint8)
        sp = json.loads(a['spine']) if isinstance(a['spine'], str) else a['spine']
        if sp and len(sp) >= 4:
            pts = np.round(np.asarray(sp, np.float64).reshape(-1, 2)).astype(np.int32)
            cv2.polylines(spine, [pts], False, 1, thickness=3)
    return full, spine


def build_one(args):
    path, recs = args
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img.shape == (H, W), path
    soft = np.zeros((H, W), np.float32)
    sp = np.zeros((H, W), np.float32)
    for anns in recs:
        f, s = record_masks(anns)
        soft += f
        sp += s
    n = max(len(recs), 1)
    return img, np.round(soft / n * 255).astype(np.uint8), np.round(sp / n * 255).astype(np.uint8), disk_stats(img)


def main(root):
    raw = os.path.join(root, 'data', 'MAGFiLO_1.0_Kaggle_2026')
    ann = json.load(open(os.path.join(root, 'data', 'annotations', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')))
    split = json.load(open(os.path.join(root, 'claude', 'configs', 'split-v2.json')))
    by_rec = {}
    for a in ann['annotations']:
        by_rec.setdefault(a['image_id'], []).append(a)
    obs = sorted(split['observations'], key=lambda o: o['file_name'])
    cache = os.path.join(root, 'cache')
    os.makedirs(cache, exist_ok=True)
    N = len(obs)
    mm = {k: np.lib.format.open_memmap(os.path.join(cache, f'train_{k}.npy'), 'w+', np.uint8, (N, H, W)) for k in ('img', 'soft', 'spine')}
    jobs = [(os.path.join(raw, 'train', 'train_images', o['file_name']), [by_rec[r] for r in o['annotator_image_ids']]) for o in obs]
    stats = []
    with ProcessPoolExecutor(8) as ex:
        for i, (img, soft, sp, st) in enumerate(ex.map(build_one, jobs, chunksize=4)):
            mm['img'][i], mm['soft'][i], mm['spine'][i] = img, soft, sp
            stats.append(st)
    for m in mm.values():
        m.flush()
    test_names = sorted(os.listdir(os.path.join(raw, 'test', 'test_images')))
    tm = np.lib.format.open_memmap(os.path.join(cache, 'test_img.npy'), 'w+', np.uint8, (len(test_names), H, W))
    tstats = []
    for i, n in enumerate(test_names):
        img = cv2.imread(os.path.join(raw, 'test', 'test_images', n), cv2.IMREAD_GRAYSCALE)
        tm[i] = img
        tstats.append(disk_stats(img))
    tm.flush()
    meta = {
        'train': [dict(file_name=o['file_name'], fold=o['fold'], group=o['group_id'], records=o['annotator_image_ids'],
                       disk_mean=s[0], disk_std=s[1], disk_thr=s[2]) for o, s in zip(obs, stats)],
        'test': [dict(file_name=n, disk_mean=s[0], disk_std=s[1], disk_thr=s[2]) for n, s in zip(test_names, tstats)],
    }
    json.dump(meta, open(os.path.join(cache, 'meta.json'), 'w'), indent=1)
    centers = {}
    for i, o in enumerate(obs):
        cs = []
        for r in o['annotator_image_ids']:
            for a in by_rec[r]:
                x, y, w, h = json.loads(a['bbox']) if isinstance(a['bbox'], str) else a['bbox']
                cs.append((x + w / 2, y + h / 2))
        centers[str(i)] = cs
    json.dump(centers, open(os.path.join(cache, 'centers.json'), 'w'))
    print('train', N, 'test', len(test_names))


if __name__ == '__main__':
    main(sys.argv[1])
