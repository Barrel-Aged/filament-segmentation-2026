"""Loader for the fold-4 holdout comb rows (work/ideas8/bundle-hold/rows/<stem>.npz, written by build_rows.py).

Each image dict has the layout of work/ideas7/exp/p1_poolxm/run_arms.py load_imgs (and xcv.py load_imgs), so a scorer
trained on the dev rows (work/ideas7/playbook/xcomb F[74] + xm columns of work/ideas7/exp/p1_poolxm/xm) applies unchanged:
    F = rows F[variant] (74 columns, dev xcomb order)  ++  xm_cols(xm, fam, ncol)
ncol: None (F[74] only), 12 (base(xm12): R34 | cnxL), 18 (R34 | cnxL | cnx), 'sym' (run_arms.xm_cols: the two maps the row
does not come from). Variants k = 0..3 differ only in the detector block (det X 45-50, det rich 51-61, d_det 63); C2-plus
averages the scorer's predictions over the four variants before selection (src/cfil/submit2.py).
fold is 4 for every image: dev-only loaders that filter fold in (0, 1, 2, 3) must not be reused for these rows.
hits / ngt are fold-4 labels: read them only for the single authorised holdout evaluation.

    import sys; sys.path.insert(0, 'D:/FilamentSegmentationKaggle/claude/work/ideas8/bundle-hold')
    import holdrows
    imgs_k = holdrows.load(variant=k, ncol='sym')          # list of 141 dicts
    q = np.mean([predict(holdrows.load(k, 'sym')) for k in range(4)], 0)   # per image, variant-averaged
"""
import glob, json, os
import numpy as np

ROOT = 'D:/FilamentSegmentationKaggle/claude'
D = ROOT + '/work/ideas8/bundle-hold/rows'
FCOLS = (['t', 'lvl', 'log_area', 'bw', 'bh', 'aspect', 'fill', 'p_mean', 'p_max', 'p_std', 'p_edge', 'persist', 'par_ratio',
          'n_child', 'depth_up', 's_mean', 's_max', 'contrast', 'i_mean', 'r_disk', 'major', 'minor', 'width', 'elong',
          'img_n50', 'img_pmean', 'sib_n', 'rank_in_img'] +
         ['i6_' + c for c in ('x_best_iou', 'x_best_score', 'x_best_det', 'x_score_ov50', 'x_n_ov30', 'x_cov_sel')] +
         ['i6_' + c for c in ('r_ov_score_max', 'r_ov_score_sum', 'r_score_ov3050', 'r_score_ov1030', 'r_best_mprob', 'r_cov_mine',
                              'r_cov_ext', 'r_inside_score', 'r_inside_n', 'r_best_rank', 'r_is_best_match')] +
         ['det_' + c for c in ('x_best_iou', 'x_best_score', 'x_best_det', 'x_score_ov50', 'x_n_ov30', 'x_cov_sel')] +
         ['det_' + c for c in ('r_ov_score_max', 'r_ov_score_sum', 'r_score_ov3050', 'r_score_ov1030', 'r_best_mprob', 'r_cov_mine',
                               'r_cov_ext', 'r_inside_score', 'r_inside_n', 'r_best_rank', 'r_is_best_match')] +
         ['d_i6', 'd_det', 'is_mg', 'is_pm', 'is_cnxL', 'is_R34'] + [f'fam_extra{i}' for i in range(6)])
XMCOLS = [f'{m}_{c}' for m in ('r34', 'cnxL', 'cnx')
          for c in ('xm_mean', 'xm_max', 'xm_frac50', 'xm_edge', 'xm_iou_same', 'xm_best_iou')]
assert len(FCOLS) == 74 and len(XMCOLS) == 18


def xm_cols(X, fam, ncol):
    """identical to work/ideas7/exp/p1_poolxm/run_arms.py xm_cols"""
    if ncol is None:
        return X[:, :0]
    if ncol != 'sym':
        return X[:, :ncol]
    A = np.where((fam == 6)[:, None], X[:, 6:12], X[:, 0:6])
    B = np.where(np.isin(fam, (5, 6))[:, None], X[:, 12:18], X[:, 6:12])
    return np.concatenate([A, B], 1)


def stems():
    return [f[:-5] for f in json.load(open(ROOT + '/work/maps/final/holdcnx.npy.json'))['files']]


def load(variant, ncol=None, with_labels=True):
    imgs = []
    for s in sorted(stems()):
        z = np.load(f'{D}/{s}.npz')
        assert bool(z['ok'])
        im = dict(name=s, fold=4, F=np.concatenate([z['F'][variant], xm_cols(z['xm'], z['fam'], ncol)], 1), fam=z['fam'],
                  parents=z['parents'], conf=z['conf'], nb=int(z['nb']), variant=variant)
        if with_labels:
            im.update(hits=z['hits'], ngt=z['ngt'])
        imgs.append(im)
    return imgs
