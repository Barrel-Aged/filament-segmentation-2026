"""Report input R4, step 2: pool the per-fold pair tables of rubric_dev.py over all dev annotator records and write the
rubric table (md + csv) and the plots (png, drawn with Pillow; matplotlib is not installed in the CPU venv).

Definitions follow the organisers' self-evaluation notebook
(D:/FilamentSegmentationKaggle/research/notebooks/official-evaluation/self-evaluation-notebook.source.txt):
  * per annotator record, IoU and Dice matrices between every GT polygon and every predicted instance of the image
    (get_overlap_matrices: IoU = |A&B| / |A|B|, Dice = 2|A&B| / (|A| + |B|));
  * PQ (get_pq_score): TP = pairs with IoU > 0.5, FP = predictions whose column has no TP, FN = GT rows with no TP,
    PQ = sum IoU(TP) / (TP + FP/2 + FN/2), pooled over all records;
  * distributions (plot_distribution): all pairs with a score > 0, pooled, 50 bins on [0, 1], probability per bin;
  * relations (plot_m2n_counts): per GT instance the number of predictions with IoU > 0 (1:n), per prediction the number
    of GT instances of the record with IoU > 0 (n:1), pooled over records (a prediction is counted once per record).
Added here (not in the notebook): the same distributions restricted to matched pairs (IoU > 0.5), many-to-many
components (connected components of the record's IoU > 0 graph with >= 2 GT and >= 2 predictions), and an image
bootstrap SE of PQ (2,000 draws, images resampled with all their records, numpy default_rng(0)).
usage: rubric_combine.py
"""
import os, json, csv
import numpy as np
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

D = 'D:/FilamentSegmentationKaggle/claude/work/package/report_inputs/rubric'
F = lambda s, b=False: ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if b else 'C:/Windows/Fonts/arial.ttf', s)


def load():
    Z = [np.load(f'{D}/pairs_f{k}.npz') for k in range(4)]
    return Z


def pq_of(tp_iou, fp, fn):
    return tp_iou.sum() / (len(tp_iou) + 0.5 * fp + 0.5 * fn)


def main():
    Z = load()
    recs = []          # per record dict
    img_key = []
    for z in Z:
        k = int(z['fold']); P = z['pairs']; ga = json.loads(str(z['gt_areas'])); pa = json.loads(str(z['pr_areas']))
        for r in range(len(z['rec_id'])):
            m = P[:, 1] == r
            p = P[m]
            inter = p[:, 4].astype(float); a = p[:, 5].astype(float); b = p[:, 6].astype(float)
            iou = inter / (a + b - inter); dice = 2 * inter / (a + b)
            recs.append(dict(fold=k, img=f'{k}:{int(z["rec_img"][r])}', team=str(z['team'][int(z['rec_img'][r])]),
                             rid=str(z['rec_id'][r]), ngt=int(z['rec_ngt'][r]), npred=int(z['rec_npred'][r]),
                             gi=p[:, 2], pj=p[:, 3], iou=iou, dice=dice))
    # PQ per record
    S = np.zeros((len(recs), 4))
    for i, r in enumerate(recs):
        hit = r['iou'] > 0.5
        tp = hit.sum()
        fp = r['npred'] - len(np.unique(r['pj'][hit]))
        fn = r['ngt'] - len(np.unique(r['gi'][hit]))
        S[i] = [r['iou'][hit].sum(), tp, fp, fn]
        r['multi_tp'] = int(tp - len(np.unique(r['pj'][hit])))
    tot = S.sum(0)
    PQ = tot[0] / (tot[1] + 0.5 * (tot[2] + tot[3]))
    SQ, RQ = tot[0] / tot[1], tot[1] / (tot[1] + 0.5 * (tot[2] + tot[3]))
    # image bootstrap
    imgs = sorted({r['img'] for r in recs})
    ix = {k: i for i, k in enumerate(imgs)}
    Simg = np.zeros((len(imgs), 4))
    for i, r in enumerate(recs):
        Simg[ix[r['img']]] += S[i]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, len(imgs), len(imgs)), minlength=len(imgs))
        c = (Simg * w[:, None]).sum(0); bs.append(c[0] / (c[1] + 0.5 * (c[2] + c[3])))
    SE = float(np.std(bs, ddof=1))
    perfold = {}
    for k in range(4):
        c = S[[r['fold'] == k for r in recs]].sum(0); perfold[k] = c[0] / (c[1] + 0.5 * (c[2] + c[3]))
    team = {}
    for name, sel in (('test teams 02/03/05/06', lambda t: t in ('02', '03', '05', '06')), ('teams 01/04', lambda t: t in ('01', '04'))):
        c = S[[sel(r['team']) for r in recs]].sum(0); team[name] = c[0] / (c[1] + 0.5 * (c[2] + c[3]))
    # analysis-scorer PQ (hyg_read semantics) from the stored counts, and the B10 validation
    CA = sum(z['C_P1019'].sum(0) for z in Z); CB = sum(z['C_B10'].sum(0) for z in Z)
    pqA = CA[0] / (CA[1] + 0.5 * (CA[2] + CA[3])); pqB = CB[0] / (CB[1] + 0.5 * (CB[2] + CB[3]))
    # distributions
    all_iou = np.concatenate([r['iou'] for r in recs]); all_dice = np.concatenate([r['dice'] for r in recs])
    m = all_iou > 0.5
    # relations
    gdeg, pdeg = Counter(), Counter()
    m2m = 0
    for r in recs:
        g = np.bincount(r['gi'], minlength=r['ngt']); p = np.bincount(r['pj'], minlength=r['npred'])
        gdeg.update(g.tolist()); pdeg.update(p.tolist())
        # many-to-many components of the bipartite overlap graph
        par = list(range(r['ngt'] + r['npred']))

        def fnd(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        for a, b in zip(r['gi'], r['pj']):
            par[fnd(int(a))] = fnd(r['ngt'] + int(b))
        comp = Counter(); compg = Counter()
        for x in range(r['ngt']):
            compg[fnd(x)] += 1
        for x in range(r['npred']):
            comp[fnd(r['ngt'] + x)] += 1
        m2m += sum(1 for c in compg if compg[c] >= 2 and comp.get(c, 0) >= 2)
    nrec = len(recs); ngt = sum(r['ngt'] for r in recs); npr = sum(r['npred'] for r in recs)
    q = lambda x: [float(np.mean(x)), float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75))]
    res = dict(images=len(imgs), records=nrec, gt_instances=ngt, prediction_record_pairs=npr,
               predictions=int(sum(z['rec_npred'][np.unique(z['rec_img'], return_index=True)[1]].sum() for z in Z)),
               PQ=PQ, PQ_bootstrap_SE=SE, SQ=SQ, RQ=RQ, TP=int(tot[1]), FP=int(tot[2]), FN=int(tot[3]),
               PQ_per_fold=perfold, PQ_by_team=team, PQ_analysis_scorer=float(pqA), counts_analysis_scorer=CA.tolist(),
               B10_S_PQ_reproduced=float(pqB), multi_match_tp=int(sum(r['multi_tp'] for r in recs)),
               overlap_pairs=int(len(all_iou)), matched_pairs=int(m.sum()),
               iou_all=q(all_iou), dice_all=q(all_dice), iou_matched=q(all_iou[m]), dice_matched=q(all_dice[m]),
               gt_degree={int(k): int(v) for k, v in sorted(gdeg.items())},
               pred_degree={int(k): int(v) for k, v in sorted(pdeg.items())}, many_to_many_components=int(m2m),
               b10_checks=[bool(z['b10_check']) for z in Z], rebuild_maxdiff=[float(z['rebuild_maxdiff']) for z in Z])
    json.dump(res, open(f'{D}/rubric_summary.json', 'w'), indent=1)
    write_tables(res)
    hist(all_iou, f'{D}/iou_hist.png', 'IoU', 'IoU of overlapping GT-prediction pairs (dev OOF)', m_iou=all_iou[m])
    hist(all_dice, f'{D}/dice_hist.png', 'Dice', 'Dice of overlapping GT-prediction pairs (dev OOF)', m_iou=all_dice[m])
    m2n(gdeg, pdeg, f'{D}/relations.png')
    print(json.dumps(res, indent=1))


def fmt(x, d=4):
    return f'{x:.{d}f}'


def write_tables(r):
    rows = [('dev images (folds 0-3)', r['images']), ('annotator records', r['records']), ('GT instances', r['gt_instances']),
            ('predicted instances', r['predictions']), ('prediction x record pairs (FP base)', r['prediction_record_pairs']),
            ('PQ (notebook definition)', fmt(r['PQ'])), ('PQ image-bootstrap SE', fmt(r['PQ_bootstrap_SE'])),
            ('SQ (mean IoU of TP pairs)', fmt(r['SQ'])), ('RQ', fmt(r['RQ'])), ('TP / FP / FN', f"{r['TP']} / {r['FP']} / {r['FN']}"),
            ('PQ per fold 0 / 1 / 2 / 3', ' / '.join(fmt(r['PQ_per_fold'][k]) for k in range(4))),
            ('PQ, test-team records (02/03/05/06)', fmt(r['PQ_by_team']['test teams 02/03/05/06'])),
            ('PQ, teams 01/04 records', fmt(r['PQ_by_team']['teams 01/04'])),
            ('PQ, analysis scorer (label-map GT, as hyg_read)', fmt(r['PQ_analysis_scorer'], 6)),
            ('overlapping pairs (IoU > 0)', r['overlap_pairs']), ('matched pairs (IoU > 0.5)', r['matched_pairs']),
            ('IoU, all overlapping pairs: mean / median / IQR', '{:.3f} / {:.3f} / {:.3f}-{:.3f}'.format(*r['iou_all'])),
            ('IoU, matched pairs: mean / median / IQR', '{:.3f} / {:.3f} / {:.3f}-{:.3f}'.format(*r['iou_matched'])),
            ('Dice, all overlapping pairs: mean / median / IQR', '{:.3f} / {:.3f} / {:.3f}-{:.3f}'.format(*r['dice_all'])),
            ('Dice, matched pairs: mean / median / IQR', '{:.3f} / {:.3f} / {:.3f}-{:.3f}'.format(*r['dice_matched'])),
            ('many-to-many components (>= 2 GT and >= 2 predictions linked at IoU > 0)', r['many_to_many_components'])]
    gd, pd = r['gt_degree'], r['pred_degree']
    def deg(c, k):
        return sum(v for d, v in c.items() if (d >= k if k == 4 else d == k))
    for k, lab in ((0, '0 (miss)'), (1, '1'), (2, '2'), (3, '3'), (4, '4 or more')):
        rows.append((f'GT instances overlapped by {lab} prediction(s) [1:n]', deg(gd, k)))
    for k, lab in ((0, '0 (miss)'), (1, '1'), (2, '2'), (3, '3'), (4, '4 or more')):
        rows.append((f'predictions overlapping {lab} GT instance(s) of a record [n:1]', deg(pd, k)))
    with open(f'{D}/rubric_table.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['measure', 'value']); w.writerows(rows)
    with open(f'{D}/rubric_table.md', 'w', encoding='utf-8') as f:
        f.write('| measure | value |\n|---|---|\n')
        for a, b in rows:
            f.write(f'| {a} | {b} |\n')
        f.write('\nFull degree counts. GT (1:n): ' + ', '.join(f'{k}: {v}' for k, v in gd.items()) +
                '. Predictions (n:1): ' + ', '.join(f'{k}: {v}' for k, v in pd.items()) + '.\n')


# ------------------------------------------------------------------ plots (Pillow)
WPX, HPX = 1000, 640
L, Rm, T, B = 110, 30, 60, 110


def axes(d, xmin, xmax, ymax, xt, yt, xl, yl, title, fmt_y):
    X = lambda v: L + (v - xmin) / (xmax - xmin) * (WPX - L - Rm)
    Y = lambda v: HPX - B - v / ymax * (HPX - T - B)
    for v in yt:
        d.line([L, Y(v), WPX - Rm, Y(v)], fill=(220, 220, 220), width=1)
        d.text((L - 10, Y(v)), fmt_y(v), fill='black', font=F(24), anchor='rm')
    d.line([L, HPX - B, WPX - Rm, HPX - B], fill='black', width=2)
    d.line([L, T, L, HPX - B], fill='black', width=2)
    for v, s in xt:
        d.line([X(v), HPX - B, X(v), HPX - B + 8], fill='black', width=2)
        d.text((X(v), HPX - B + 12), s, fill='black', font=F(24), anchor='mt')
    d.text(((L + WPX - Rm) / 2, HPX - 30), xl, fill='black', font=F(26), anchor='mb')
    d.text((L, 30), title, fill='black', font=F(26, True), anchor='lm')
    img_y = Image.new('RGBA', (300, 40), (255, 255, 255, 0))
    ImageDraw.Draw(img_y).text((150, 20), yl, fill='black', font=F(26), anchor='mm')
    return X, Y, img_y.rotate(90, expand=True)


def hist(x, path, xl, title, m_iou):
    bins = np.linspace(0, 1, 51)
    c, _ = np.histogram(x, bins); p = c / c.sum()
    cm, _ = np.histogram(m_iou, bins); pm = cm / c.sum()
    ymax = float(np.ceil(p.max() * 50) / 50 + 0.02)
    img = Image.new('RGB', (WPX, HPX), 'white'); d = ImageDraw.Draw(img)
    yt = list(np.arange(0, ymax + 1e-9, 0.02 if ymax <= 0.12 else 0.05))
    X, Y, ylab = axes(d, 0, 1, ymax, [(v, f'{v:.1f}') for v in np.arange(0, 1.01, 0.2)], yt, xl, 'Probability', title,
                      lambda v: f'{v:.2f}')
    img.paste(ylab, (5, (HPX - ylab.height) // 2 - 20), ylab)
    for i in range(50):
        x0, x1 = X(bins[i]) + 1, X(bins[i + 1]) - 1
        d.rectangle([x0, Y(p[i]), x1, Y(0)], fill=(170, 185, 215), outline=(60, 60, 60))
        if pm[i] > 0:
            d.rectangle([x0, Y(pm[i]), x1, Y(0)], fill=(40, 70, 140), outline=(30, 30, 30))
    mu_ = float(x.mean())
    d.line([X(mu_), T, X(mu_), Y(0)], fill=(200, 20, 60), width=3)
    d.text((X(mu_) + 8, T + 30), f'mean {mu_:.3f} (n = {len(x):,})', fill=(200, 20, 60), font=F(24), anchor='lm')
    d.text((X(mu_) + 8, T + 62), f'dark: IoU > 0.5 pairs ({len(m_iou):,})', fill=(40, 70, 140), font=F(24), anchor='lm')
    img.save(path, dpi=(300, 300))


def m2n(gdeg, pdeg, path):
    cap = 4
    g = Counter(); p = Counter()
    for k, v in gdeg.items():
        g[min(k, cap)] += v
    for k, v in pdeg.items():
        p[min(k, cap)] += v
    ymax = max(list(g.values()) + list(p.values()))
    ymax = float(np.ceil(ymax / 1000) * 1000)
    img = Image.new('RGB', (WPX, HPX), 'white'); d = ImageDraw.Draw(img)
    pos = [(-(k + 0.5), f'{k}:1' if k < cap else f'{cap}+:1') for k in range(cap, -1, -1)] + \
          [(k + 0.5, f'1:{k}' if k < cap else f'1:{cap}+') for k in range(cap + 1)]
    X, Y, ylab = axes(d, -cap - 1.1, cap + 1.1, ymax, pos, list(np.arange(0, ymax + 1, 1000)), 'Matching case (IoU > 0)',
                      'Frequency', 'One-to-many and many-to-one relations (dev OOF)', lambda v: f'{int(v):,}')
    img.paste(ylab, (5, (HPX - ylab.height) // 2 - 20), ylab)
    d.line([X(0), T, X(0), Y(0)], fill='black', width=2)
    for k in range(cap + 1):
        for v, xc, col in ((p.get(k, 0), -(k + 0.5), (255, 140, 0) if k == 0 else (70, 130, 180)),
                           (g.get(k, 0), k + 0.5, (220, 20, 60) if k == 0 else (34, 139, 34))):
            d.rectangle([X(xc - 0.4), Y(v), X(xc + 0.4), Y(0)], fill=col, outline=(30, 30, 30))
            d.text((X(xc), Y(v) - 4), f'{v:,}', fill='black', font=F(20), anchor='mb')
    d.text((X(-cap - 1), T + 40), 'predictions (n:1)', fill='black', font=F(24, True), anchor='lm')
    d.text((X(cap + 1), T + 40), 'GT instances (1:n)', fill='black', font=F(24, True), anchor='rm')
    img.save(path, dpi=(300, 300))


if __name__ == '__main__':
    main()
