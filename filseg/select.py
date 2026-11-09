"""Stage 7: the row score s and the exact selection at tau 0.24.

For each image, every combS row gets
    s = q - 0.2 * |t - 0.475| - 0.03 * [family != 0]
where t is the row's threshold level (column 0 of the scorer columns). The selection keeps the set of rows with
s > tau that maximises sum(s - tau) subject to no two kept rows conflicting (two rows conflict when their masks
overlap; stage 5 lists the pairs in ``conf``). Rows with s <= tau never help, so they are dropped first. The conflict
graph of the remaining rows splits into connected components; a single-row component is kept as is, and each larger
one is solved as a 0/1 integer program (one variable per row, one ``x_a + x_b <= 1`` per conflict) with
``scipy.optimize.milp`` (HiGHS).

Changes from the original run
- ``select`` is ``xcv.select`` (work/ideas7/playbook/xcv.py:53-81) without its first branch, the tree DP for the
  "base" layout (xcv.py:54-55), which the combS pool never takes. ``select_image`` is ``score_select.select_one``
  (work/ideas8/bundle-test/score_select.py:59-65) with the configuration fixed to the final one,
  ``('combS', 0.475, 0.2, 0.03, 0.24)`` (score_select.py:37, csv_score.py:32).
- Importing xcv had side effects (argv parsing, a patch of ``cfil.stack.gbm_params``, and ``cfil.decide`` reading the
  split file; xcv.py:19-27). The selection itself needs only numpy and scipy; the command line reads the rows
  through filseg.score.
- ``select`` takes two optional arguments the original did not have. ``options`` is passed to ``milp``; the default
  None makes the same call as xcv.py:78-79 (HiGHS defaults, so the relative MIP gap tolerance is HiGHS's default, not
  0). ``results``, if given a list, receives every ``milp`` result so a caller can check the solver status and gap.
- ``select`` raises ``RuntimeError`` unless ``milp`` reports status 0 (optimal). The original had no such check. It
  would keep a feasible solution that was not proven optimal, and it would fail with a ``TypeError`` on ``r.x`` when
  there was no solution.
- ``save`` writes the per-image selections with the layout of the ``'S1'`` entry of the original ``sel_s1.pkl``
  (csv_score.py:114-122) to a path you choose; the provenance fields of that file (q707 and ET training-set hashes,
  note) are replaced by the sha256 of the scores file.
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from filseg import paths
from filseg.score import load_views

T0, LAMBDA, DELTA, TAU = 0.475, 0.2, 0.03, 0.24
CFG = ("combS", T0, LAMBDA, DELTA, TAU)


def row_score(q: np.ndarray, sub: dict) -> np.ndarray:
    """s = q - 0.2 |t - 0.475| - 0.03 [family != 0] for the rows of one image (score_select.select_one)."""
    q = np.asarray(q, np.float64)
    assert len(q) == len(sub["F"])
    t = sub["F"][:, 0].astype(np.float64); new = (sub["fam"] != 0).astype(np.float64)
    return q - LAMBDA * np.abs(t - T0) - DELTA * new


def _solve_component(w: np.ndarray, ea: np.ndarray, eb: np.ndarray, options: dict | None, results: list | None):
    """Max sum(w x) over x in {0, 1}^n with x[ea[e]] + x[eb[e]] <= 1 for every edge e; returns the 0/1 vector."""
    n = len(w)
    A = coo_matrix((np.ones(2 * len(ea)), (np.repeat(np.arange(len(ea)), 2), np.stack([ea, eb], 1).ravel())),
                   shape=(len(ea), n))
    r = milp(c=-w, constraints=LinearConstraint(A, -np.inf, 1), integrality=np.ones(n), bounds=Bounds(0, 1),
             options=options)
    if results is not None:
        results.append(r)
    if r.status != 0:
        raise RuntimeError(f"milp did not prove an optimal solution: status {r.status}, {r.message}")
    return r.x > 0.5


def select(sub: dict, s: np.ndarray, tau: float = TAU, options: dict | None = None,
           results: list | None = None) -> np.ndarray:
    """Indices (into the image's sub-rows) of the conflict-free set maximising sum(s - tau) (xcv.select)."""
    w = s - tau
    pos = np.flatnonzero(w > 0)
    if not len(pos):
        return pos
    loc = -np.ones(len(s), np.int64); loc[pos] = np.arange(len(pos))
    c = sub["conf"]
    c = c[(loc[c[:, 0]] >= 0) & (loc[c[:, 1]] >= 0)]
    if not len(c):
        return pos
    a, b = loc[c[:, 0]], loc[c[:, 1]]
    G = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(pos), len(pos)))
    ncc, lab = connected_components(G, directed=False)
    out = []
    for cc in range(ncc):
        nodes = np.flatnonzero(lab == cc)
        if len(nodes) == 1:
            out.append(pos[nodes[0]]); continue
        ln = -np.ones(len(pos), np.int64); ln[nodes] = np.arange(len(nodes))
        e = (ln[a] >= 0)
        keep = _solve_component(w[pos[nodes]], ln[a[e]], ln[b[e]], options, results)
        out.extend(pos[nodes[keep]])
    return np.asarray(out, np.int64)


def select_image(sub: dict, q: np.ndarray, options: dict | None = None, results: list | None = None) -> np.ndarray:
    """The selected sub-rows of one image at the final configuration (score_select.select_one)."""
    s = row_score(q, sub)
    return np.asarray(select(sub, s, TAU, options, results), np.int64) if len(s) else np.zeros(0, np.int64)


def select_all(subs: list[dict], qs: list[np.ndarray], options: dict | None = None,
               results: list | None = None) -> dict:
    """``sel_sub`` (sub-row indices) and ``sel_rows`` (row-file indices) per image name (csv_score.py:97, 114)."""
    sel = {im["name"]: select_image(im, qq, options, results) for im, qq in zip(subs, qs)}
    return dict(sel_sub=sel, sel_rows={im["name"]: im["idx"][sel[im["name"]]] for im in subs}, cfg=CFG)


def save(path: Path, res: dict, scores_sha256: str) -> None:
    """Write ``{'sel': {'S1': res}, 'scores_sha256': ...}`` as a pickle (the layout of the original sel_s1.pkl)."""
    with open(path, "wb") as f:
        pickle.dump(dict(sel={"S1": res}, scores_sha256=scores_sha256), f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select the conflict-free rows at tau 0.24 from the scores.")
    ap.add_argument("--rows", type=Path, default=paths.WORK / "rows", help="folder of row files (stage 5)")
    ap.add_argument("--scores", type=Path, default=paths.WORK / "scores.npz", help="output of filseg.score")
    ap.add_argument("--out", type=Path, default=None, help="output .pkl (default <FILSEG_WORK>/selection.pkl)")
    a = ap.parse_args()
    subs = load_views(a.rows)[0]
    z = np.load(a.scores, allow_pickle=True)
    assert list(z["names"]) == [im["name"] for im in subs]
    assert all(np.array_equal(np.asarray(r), im["idx"]) for r, im in zip(z["rows"], subs))
    res = select_all(subs, list(z["q"]))
    out = a.out or paths.work("selection.pkl")
    save(out, res, hashlib.sha256(Path(a.scores).read_bytes()).hexdigest())
    print(f"selected {sum(len(v) for v in res['sel_sub'].values())} rows on {len(subs)} images -> {out}")


if __name__ == "__main__":
    main()
