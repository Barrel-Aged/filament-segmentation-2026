"""Report input R4: rubric diagnostics of the shipped configuration (S1-S at tau 0.24) on dev out-of-fold predictions.

One dev fold per process.  CPU only.  Read-only on every original file; writes only under
D:/FilamentSegmentationKaggle/claude/work/package/report_inputs/rubric/.

What it rebuilds (by importing the frozen ideas15 analysis module work/ideas15/hyg/hyg_read.py, sha256 ac79cb81..., as a
module; nothing in it is modified and its main() is not called):
  * the combS candidate pool of every dev image of the fold from the stored OOF maps (cnx, cnxL, R34) and the stored combS
    rows (hyg_read.rebuild; every rebuilt row is checked against the stored hits and features, as hyg_read does);
  * the B3f decision: s = q - 0.2|t - 0.475| - 0.03[fam != 0], exact ILP selection at tau 0.24 (xcv.select);
  * arm S: the 0.23 px shift toward row 0 of every selected family 0/5/6 row on its own OOF map, then pixel claiming in
    descending s (hyg_read.reshape / claim_rows).
Score sets (q, cross-fitted by dev fold, stored in work/ideas14/gate/q/repro):
  * 'P1019' (the analysed set): 0.7 x mean of LightGBM seeds 10-19 + 0.3 x mean of ExtraTrees seeds 10-19, i.e. the dev
    analogue of the shipped q (production: same seeds, same 0.7/0.3 blend; on test the LightGBM half was refit on 707 images).
  * 'B10' (validation only): the recorded secondary set (seeds 10-14); its per-image [S, TP, FP, FN] under arm S must equal
    work/ideas15/hyg/runs/sec_f<k>.npz C__B10__S exactly.
Metrics on the final masks, with the organisers' self-evaluation notebook definitions
(research/notebooks/official-evaluation/self-evaluation-notebook.source.txt, get_overlap_matrices / get_pq_score /
plot_distribution / plot_m2n_counts): per annotator record, the full (n_gt x n_pred) IoU and Dice matrices between every GT
polygon (rasterised alone with pycocotools, as the notebook's per-layer decode) and every predicted instance.  This script
stores every non-zero pair; the combine step builds the histograms and relation counts.
usage: rubric_dev.py <fold>
"""
import os, sys, json, time, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.dont_write_bytecode = True
T0 = time.time()
ROOT = 'D:/FilamentSegmentationKaggle/claude'
HYG = ROOT + '/work/ideas15/hyg'
OUT = ROOT + '/work/package/report_inputs/rubric'
ANN = ROOT + '/../data/annotations/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
SPLIT = ROOT + '/configs/split-v2.json'
HYG_READ_SHA = 'ac79cb81eef77a5fdb7d076e13a30c66b022da183a16bdb6715eb3a8acf2f20d'
FOLD = int(sys.argv[1])
assert FOLD in (0, 1, 2, 3)
sys.path.insert(0, HYG)
import hyg_read as HR  # noqa: E402
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'      # hyg_read sets '' on import; -1 is what hides the GPU on this PC
assert 'torch' not in sys.modules
import numpy as np  # noqa: E402
from pycocotools import mask as mu  # noqa: E402

H = W = 2048
HR.SETS['P1019'] = ('b7', list(range(10, 20)), 'repro', False)
SETLIST = ['P1019', 'B10']
ARM = 'S'


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def enc(Q):
    m = np.zeros(H * W, np.uint8); m[Q] = 1
    r = mu.encode(np.asfortranarray(m.reshape(H, W)))
    return r['counts'].decode('ascii')


def main():
    assert sha(HR.__file__) == HYG_READ_SHA, 'hyg_read.py changed'
    assert HR.ARMS[ARM] == (False, (-0.23, 0.0), (0, 5, 6), False) and HR.TAU == 0.24
    imgs = HR.SC.load()
    Q, qfiles = HR.load_sets(imgs, SETLIST)
    n = np.array([len(im['F']) for im in imgs]); off = np.concatenate([[0], np.cumsum(n)])
    split = json.load(open(SPLIT))
    obs = [o for o in split['observations'] if o['fold'] == FOLD]
    recs_of = {o['file_name'][:-5]: o['annotator_image_ids'] for o in obs}
    keep = {r for o in obs for r in o['annotator_image_ids']}
    by = {}
    for a in json.load(open(ANN))['annotations']:
        if a['image_id'] in keep:
            by.setdefault(a['image_id'], []).append(a)
    maps = {}
    for name in ('cnx', 'cnxL', 'r34'):
        mp = ROOT + f'/work/maps/{name}-f{FOLD}/oof.npy'
        maps[name] = (np.load(mp, mmap_mode='r'), {f[:-5]: j for j, f in enumerate(json.load(open(mp + '.json'))['files'])})
    gidx = [i for i, im in enumerate(imgs) if im['fold'] == FOLD]
    ref = np.load(f'{HYG}/runs/sec_f{FOLD}.npz')
    assert list(ref['names']) == [imgs[i]['name'] for i in gidx]
    ni = len(gidx)
    C = {s_: np.zeros((ni, 4)) for s_ in SETLIST}
    claim = np.zeros(H * W, bool)
    csv_rows = []                       # P1019 final masks: (image stem, instance index, rle, fam, s, area)
    pairs = []                          # (image idx, record idx, gt j, pred k, inter, gt area, pred area)
    recs = []                           # (image idx, record id, n_gt, n_pred)
    gt_areas, pr_areas = [], []         # per record: list of GT areas / pred areas (for zero-degree counts)
    maxdh = 0.0
    for ii, i in enumerate(gidx):
        t1 = time.time()
        im = imgs[i]; name = im['name']
        P = np.asarray(maps['cnx'][0][maps['cnx'][1][name], 0])
        PL = np.asarray(maps['cnxL'][0][maps['cnxL'][1][name], 0])
        PR = np.asarray(maps['r34'][0][maps['r34'][1][name], 0])
        own = {0: P, 2: P, 4: P, 5: PL, 6: PR}
        gts = HR.gt_label_maps(by, recs_of[name]); R = len(gts)
        sets, lev = HR.rebuild(im, P, PL, PR)
        if len(sets):
            Hs = np.stack([HR.hits_of(Qr, gts) for Qr in sets])
            d = float(np.max(np.abs(Hs - im['hits'])))
            HR.need(d <= 1e-5, name, 'rebuilt hits differ from stored hits', d)
            maxdh = max(maxdh, d)
            HR.feature_check(im, sets, lev, own)
        t = im['F'][:, 0].astype(np.float64); fam = im['fam']; lvl = im['F'][:, 1].astype(np.int64)
        ngt = float(im['ngt'].sum())
        for sid in SETLIST:
            qi = Q[sid][off[i]:off[i + 1]]
            s = qi - 0.2 * np.abs(t - 0.475) - 0.03 * (fam != 0)
            sel = [int(r) for r in HR.xcv.select(im, s, HR.TAU)]
            guard, sh, fams, compat = HR.ARMS[ARM]
            masks = {r: (HR.reshape(sets[r], own[fam[r]], HR.X.THR[lvl[r]], sh) if fam[r] in fams else sets[r]) for r in sel}
            kept, nlp, nlc, ner = HR.claim_rows(HR.claim_order(sel, s), masks, claim)
            hh = np.stack([HR.hits_of(Qk, gts).astype(np.float64) for _, Qk in kept]) if kept else np.zeros((0, R))
            tp = float((hh > 0).sum())
            C[sid][ii] = [hh.sum(), tp, len(kept) * R - tp, ngt - tp]
            if sid != 'P1019':
                continue
            # final instances of the analysed set, in claiming order
            plab = np.zeros(H * W, np.int32)
            pa = []
            for k, (r, Qk) in enumerate(kept, 1):
                plab[Qk] = k; pa.append(len(Qk))
                csv_rows.append((name, k, enc(Qk), int(fam[r]), float(s[r]), len(Qk)))
            pa = np.asarray(pa, np.int64)
            for ri, rid in enumerate(recs_of[name]):
                ga = []
                for j, a in enumerate(by[rid]):
                    segs = json.loads(a['segmentation']) if isinstance(a['segmentation'], str) else a['segmentation']
                    m = mu.decode(mu.merge(mu.frPyObjects([sg for sg in segs if len(sg) >= 6], H, W))).astype(bool).ravel()
                    ga.append(int(m.sum()))
                    cnt = np.bincount(plab[m], minlength=len(pa) + 1)
                    for k in np.flatnonzero(cnt[1:]) + 1:
                        pairs.append((ii, len(recs), j, k - 1, int(cnt[k]), ga[-1], int(pa[k - 1])))
                recs.append((ii, rid, len(ga), len(pa)))
                gt_areas.append(ga); pr_areas.append(pa.tolist())
        if ii % 20 == 0:
            print(f'fold {FOLD} image {ii + 1}/{ni} {name} rows {len(sets)} R {R} ({time.time() - t1:.1f}s, elapsed '
                  f'{time.time() - T0:.0f}s)', flush=True)
    okB = bool(np.array_equal(C['B10'][:, 1:], ref['C__B10__S'][:, 1:]) and
               np.abs(C['B10'][:, 0] - ref['C__B10__S'][:, 0]).max() <= 1e-4)
    pq = lambda c: c[0] / (c[1] + 0.5 * (c[2] + c[3]))
    print(f'fold {FOLD}: rebuild max |hits - stored| {maxdh:.2e}; B10/S per-image counts == sec_f{FOLD}.npz C__B10__S: {okB} '
          f'(PQ {pq(C["B10"].sum(0)):.6f} vs {pq(ref["C__B10__S"].sum(0)):.6f}); P1019/S analysis PQ {pq(C["P1019"].sum(0)):.6f} '
          f'{C["P1019"].sum(0).round(3).tolist()}', flush=True)
    assert okB, 'validation against the recorded B10 arm-S read failed'
    os.makedirs(OUT, exist_ok=True)
    with open(f'{OUT}/dev_oof_S1S_t024_f{FOLD}.csv', 'w', encoding='utf-8', newline='') as fh:
        fh.write('filament_id,segmentation_rle\n')
        for stem, k, rle, _, _, _ in csv_rows:
            fh.write(f'{stem}_{k},{rle}\n')
    pr = np.asarray(pairs, np.int64).reshape(-1, 7)
    np.savez(f'{OUT}/pairs_f{FOLD}.npz', fold=FOLD, names=np.array([imgs[i]['name'] for i in gidx]),
             team=ref['team'], pairs=pr, rec_img=np.array([r[0] for r in recs]), rec_id=np.array([r[1] for r in recs]),
             rec_ngt=np.array([r[2] for r in recs]), rec_npred=np.array([r[3] for r in recs]),
             gt_areas=json.dumps(gt_areas), pr_areas=json.dumps(pr_areas),
             inst=json.dumps([(a, b, c, d, e) for a, b, _, c, d, e in csv_rows]),
             C_P1019=C['P1019'], C_B10=C['B10'], qfiles=json.dumps(qfiles), hyg_read_sha=sha(HR.__file__),
             script_sha=sha(os.path.abspath(__file__)), b10_check=okB, rebuild_maxdiff=maxdh)
    print(f'wrote fold {FOLD}: {len(csv_rows)} instances, {len(recs)} records, {len(pr)} overlapping pairs; '
          f'{time.time() - T0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
