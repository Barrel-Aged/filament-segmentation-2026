"""The 0.23 px shift S and claiming: the last change made to the selected masks before they are written. CPU only.

What S does to one selected row of family 0 (cnx tree node), 5 (cnxL tree node) or 6 (R34 tree node)
1. Crop the row's own probability map (channel 0 of the 4-fold test average: ``testcnx`` for family 0, ``testcnxL`` for
   5, ``testr34`` for 6; uint8 0-255) to the stored mask's bounding box plus 4 px on each side (clipped to the image).
2. ``scipy.ndimage.shift(crop as float32, (-0.23, 0.0), order=1, mode='nearest')``: each pixel becomes
   P[y] + 0.23 (P[y+1] - P[y]), so the map content moves 0.23 px toward row 0. No column shift.
3. Keep the pixels strictly above THR = round(255 t), t the row's decimal threshold level (``LEVELS``), and only inside
   the 3 x 3 dilation of the stored mask. A pixel whose shifted value equals THR exactly is excluded.
Rows of family 2 (mg) and 4 (pm) are merges, not single threshold components; they keep their stored masks.

Claiming: the selected rows of an image are visited in descending s (ties by row index); each keeps its (shifted or
stored) mask minus the pixels already taken by an earlier row, so the instances stay disjoint. A row left empty is
dropped. On the final (tau 0.24) no pixel was lost to claiming. The CSV keeps the selection order of the instances, so
instance n of the shifted CSV is the shifted mask of the row behind instance n of the unshifted CSV.

Numerical notes (keep them): the shift runs in float32, as in the original; in float64 an exact tie such as
92 + 0.23 x 100 comes out as 115 + 1.8e-12 and would be included. THR must come from the decimal float64 level
(round(255 x 0.30) = 76); the float32 copy of the level stored in the feature table gives 77.

Changes from the original run
- ``reshape``, ``claim_order`` and ``claim_rows`` are copied unchanged from ``work/ideas15/hyg/hyg_read.py`` L379-417
  (only the literal crop padding 4 is now the constant ``PAD = 4``);
  ``THR`` from ``work/ideas7/playbook/xcands.py`` L30, ``LEVELS`` from ``src/cfil/cands.py`` L15, ``K3`` and ``LEV32``
  from hyg_read.py L188 and L190, the arm S constants ``(-0.23, 0.0)`` on families ``(0, 5, 6)`` from hyg_read.py L71.
  hyg_read.py's other 600 lines (the dev read, the freeze check, label maps) are not needed and not copied, and the
  module is no longer imported as a whole, which removes its hard-coded ROOT (L47), its ``sys.path`` inserts and its
  import chain (scommon, run_arms, compare, xcv, xcands, xcands2, cfil.cands/decide/stack/build_cands/rich).
- The per-image loop of ``work/ideas15/hyg/csv/patch_test.py`` L210-269 is split into small functions:
  ``row_scores`` (L225-227), ``own_maps`` (L220-222), ``image_masks`` (L234-245, L266-268) and ``direction_check``
  (L246-265). Same arithmetic, same order.
- Removed from patch_test.py: the dev-label gate (L161-176, it refused to run without the e2e_dev / e2e_diag results),
  the hash check of the frozen hyg_read.py and bt_common.py copies (L177-178), the B3f707 twin variant (VARIANTS L65-74;
  only the final's selection is processed), the hashing of the 4.5 GB maps and the manifests (L419-479), and the deletion
  and rewrite of the frozen outputs (L287-291, L407, L474, L479). The xcv.select re-run on the saved q (L228-233) belongs
  to the selection stage and is not repeated here.
- Release order: the original wrote the unshifted CSV first and patched it later (two scripts, two CSVs). Here the shift
  is applied to the selected masks before the single write. ``image_masks(..., shift=None)`` keeps the unshifted path:
  it returns the stored masks unchanged, with no reshape and no claiming, which is what the original writer wrote.
- The shared claim buffer (patch_test.py L203, used by both variants) is now created per call unless the caller passes
  one; ``claim_rows`` leaves it all False on exit in both cases, so the result is the same.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

H = W = 2048
LEVELS = np.round(np.arange(0.10, 0.851, 0.05), 3)       # decimal threshold levels t (float64)
THR = [int(round(t * 255)) for t in LEVELS]               # THR[level] = round(255 t), half to even, from float64 t
LEV32 = np.asarray(LEVELS, np.float32)                    # the float32 copy stored in the feature table F[:, 0]
K3 = np.ones((3, 3), np.uint8)
PAD = 4
SHIFT = (-0.23, 0.0)                                      # (dy, dx) in pixels; negative dy = toward row 0
SHIFT_FAMILIES = (0, 5, 6)
MAP_OF_FAMILY = {0: "cnx", 2: "cnx", 4: "cnx", 5: "cnxL", 6: "r34"}
MAP_NAMES = ("cnx", "cnxL", "r34")


# ---- maps ---------------------------------------------------------------------------------------------------------

def open_maps(map_dir, files: list) -> dict:
    """Memory-map ``test{cnx,cnxL,r34}.npy`` (180 x 2 x 2048 x 2048 uint8) and check each sidecar's file order."""
    maps = {}
    for n in MAP_NAMES:
        p = Path(map_dir) / f"test{n}.npy"
        a = np.load(p, mmap_mode="r")
        order = json.loads(Path(str(p) + ".json").read_text())["files"]
        if a.shape != (len(files), 2, H, W) or a.dtype != np.uint8 or order != list(files):
            raise ValueError(f"{p}: shape {a.shape} {a.dtype} or file order differs from the test inventory")
        maps[n] = a
    return maps


def own_maps(maps: dict, j: int) -> dict:
    """{family: channel 0 of that family's own map for image j}."""
    P, PL, PR = (np.asarray(maps[n][j, 0]) for n in MAP_NAMES)
    return {0: P, 2: P, 4: P, 5: PL, 6: PR}


# ---- the frozen functions (hyg_read.py L379-417) ------------------------------------------------------------------

def reshape(Q, Pm, thr, sh, h=H, w=W):
    """fam 0/5/6 reshape on the row's own map (h x w); returns sorted flat indices."""
    y, x = Q // w, Q % w
    y0, y1, x0, x1 = y.min(), y.max() + 1, x.min(), x.max() + 1
    y0, y1, x0, x1 = max(y0 - PAD, 0), min(y1 + PAD, h), max(x0 - PAD, 0), min(x1 + PAD, w)
    m0 = np.zeros((y1 - y0, x1 - x0), np.uint8); m0[y - y0, x - x0] = 1
    crop = Pm[y0:y1, x0:x1].astype(np.float32)
    shc = ndi.shift(crop, sh, order=1, mode="nearest")
    m = (shc > thr) & (cv2.dilate(m0, K3) > 0)
    ys, xs = np.nonzero(m)
    return ((ys + y0) * w + xs + x0).astype(np.int64)


def claim_order(sel, s):
    """claiming order: descending s, ties by row index"""
    return sorted((int(r) for r in sel), key=lambda x: (-s[x], x))


def claim_rows(order, masks, claim):
    """Each row in `order` keeps masks[r] minus pixels already claimed by an earlier row. claim: bool buffer over the flat
    image, all False on entry and on exit. Returns kept [(r, Qk)] (non-empty only) and counters (rows that lost >= 1
    pixel, rows emptied by claiming, rows whose mask was already empty)."""
    kept = []
    nlp = nlc = ner = 0
    try:
        for r in order:
            Qr = masks[r]
            er = len(Qr) == 0
            Qk = Qr[~claim[Qr]]
            nlp += len(Qk) < len(Qr); ner += er
            if len(Qk) == 0:
                nlc += not er
                continue                         # emptied row: dropped (not a prediction)
            claim[Qk] = True
            kept.append((r, Qk))
    finally:
        for _, Qk in kept:
            claim[Qk] = False
    return kept, nlp, nlc, ner


# ---- per image ----------------------------------------------------------------------------------------------------

def row_scores(q_sub, idx, t32, fam):
    """s = q - 0.2 |t - 0.475| - 0.03 [fam != 0] for the scored rows ``idx`` (NaN for the others).

    q_sub: the scorer's q of rows ``idx``; t32: the float32 level column F[:, 0] of every row, as the selection used it.
    """
    idx = np.asarray(idx, np.int64)
    tt = np.asarray(t32)[idx].astype(np.float64); new = (np.asarray(fam)[idx] != 0).astype(np.float64)
    s_sub = np.asarray(q_sub, np.float64) - 0.2 * np.abs(tt - 0.475) - 0.03 * new
    s_row = np.full(len(fam), np.nan); s_row[idx] = s_sub
    return s_row


def new_counters() -> dict:
    """Counters that ``image_masks`` adds to (summed over the images)."""
    return dict(selected=0, kept=0, changed=0, rows_lost_px=0, lost_px=0, emptied_by_claim=0, emptied_by_reshape=0,
                reshaped=0, above_thr=0, zero_identity=0)


def image_masks(sets, fam, lvl, sel_rows, s_row=None, own=None, shift=SHIFT, families=SHIFT_FAMILIES,
                counters=None, claim=None):
    """Final masks of one image's selected rows, in selection order (None for a row emptied by claiming).

    sets: pixel sets of every row; fam, lvl: family and level index of every row; sel_rows: the selection;
    s_row: row scores (``row_scores``) for the claiming order; own: ``own_maps`` of this image.
    shift=None: the unshifted path, the stored masks unchanged (no map, score or claiming needed).
    With ``counters`` (``new_counters``) it also counts the checks of the original run: stored mask strictly above THR
    on its own map, zero shift returns the stored mask, rows changed, pixels lost to claiming.
    """
    sel = [int(r) for r in sel_rows]
    if shift is None:
        return [sets[r] for r in sel]
    lvl = np.asarray(lvl).astype(np.int64)
    masks = {}
    c = counters if counters is not None else new_counters()
    for r in sel:
        if fam[r] in families:
            Pm = own[int(fam[r])]; thr = THR[lvl[r]]
            c["above_thr"] += int(Pm.ravel()[sets[r]].min()) > thr
            c["zero_identity"] += np.array_equal(reshape(sets[r], Pm, thr, (0.0, 0.0)), sets[r])
            c["reshaped"] += 1
            masks[r] = reshape(sets[r], Pm, thr, shift)
        else:
            masks[r] = sets[r]
    buf = claim if claim is not None else np.zeros(H * W, bool)
    kept, nlp, nlc, ner = claim_rows(claim_order(sel, s_row), masks, buf)
    kd = dict(kept)
    c["selected"] += len(sel); c["kept"] += len(kept)
    c["changed"] += len(sel) - len(kept) + sum(not np.array_equal(kd[r], sets[r]) for r in kd)
    c["rows_lost_px"] += nlp; c["emptied_by_claim"] += nlc; c["emptied_by_reshape"] += ner
    c["lost_px"] += sum(len(masks[r]) - len(kd[r]) for r in kd)
    return [kd.get(r) for r in sel]


def direction_check(stored, final, Pi, d=None):
    """Count changed pixels consistent with content moving toward row 0 on the row's own map ``Pi``.

    A gained pixel needs P[y+1] > P[y], a lost one P[y+1] < P[y]. Adds to ``d`` (created if None) and returns it;
    the alternatives (toward row 2047, along the columns) are counted for comparison. From patch_test.py L246-265.
    """
    if d is None:
        d = dict(instances=0, inst_all_up=0, changed_px=0, edge=0, up_ok=0, down_ok=0, colp_ok=0, colm_ok=0)
    Gp = np.setdiff1d(final, stored, assume_unique=True); Lp = np.setdiff1d(stored, final, assume_unique=True)
    d["instances"] += 1; ok_i = True
    for pix, sg in ((Gp, 1), (Lp, -1)):
        y, x = pix // W, pix % W
        e = (y + 1 >= H) | (y - 1 < 0) | (x + 1 >= W) | (x - 1 < 0)
        d["edge"] += int(e.sum()); d["changed_px"] += int(len(pix))
        y, x = y[~e], x[~e]
        p0 = Pi[y, x].astype(np.int32)
        up = np.sign(Pi[y + 1, x].astype(np.int32) - p0) == sg
        d["up_ok"] += int(up.sum())
        d["down_ok"] += int((np.sign(Pi[y - 1, x].astype(np.int32) - p0) == sg).sum())
        d["colp_ok"] += int((np.sign(Pi[y, x + 1].astype(np.int32) - p0) == sg).sum())
        d["colm_ok"] += int((np.sign(Pi[y, x - 1].astype(np.int32) - p0) == sg).sum())
        ok_i &= bool(up.all()) and not e.any()
    d["inst_all_up"] += ok_i
    return d
