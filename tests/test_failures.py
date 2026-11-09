"""Tests that a failed reproduction stops with an error instead of finishing quietly.

Run this file with plain Python: ``python tests/test_failures.py``. It takes a few seconds on the CPU. The tests build
small synthetic fixtures, so they need no competition data, no weights and no GPU.
"""
import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # An empty string doesn't hide the GPUs on Windows, but -1 does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from pycocotools import mask as mu  # noqa: E402

from filseg import pipeline, select, submit  # noqa: E402
from filseg import shift as S  # noqa: E402
from tools import verify_cpu_chain  # noqa: E402

OTHER_SHA = "0" * 64
IMAGE = "synthetic.jpeg"


def overlapping_csv() -> bytes:
    """A well-formed submission CSV with two identical instances, which breaks the rule that instances are disjoint."""
    m = np.zeros((S.H, S.W), np.uint8)
    m[10:20, 10:20] = 1
    counts = mu.encode(np.asfortranarray(m))["counts"].decode("ascii")
    stem = Path(IMAGE).stem
    return f"filament_id,segmentation_rle\n{stem}_1,{counts}\n{stem}_2,{counts}\n".encode("ascii")


def run_pipeline(tmp, mode, sha, csv_bytes=b"", validate=None):
    """Call ``pipeline.run`` with every stage replaced, except stage 10 when ``validate`` is None.

    Returns the result of the run and everything it printed. Exceptions from the run pass through to the caller.
    """
    def write(final, out_csv):
        Path(out_csv).write_bytes(csv_bytes)
        return dict(sha256=sha, instance_rows=2)

    stages = dict(stage1_images=lambda *a: ([], None), stage2_dense=lambda *a: {}, stage3_det=lambda *a: [],
                  stage4_i6=lambda *a: tmp, stage_replay=lambda *a: ({}, [], tmp), stage5a_evidence=lambda *a: {},
                  stage5b_rows=lambda *a: tmp, stage6_score=lambda *a: {}, map_files=lambda *a: [],
                  stage7_select=lambda *a: {}, stage9_write=write,
                  stage8_shift=lambda *a: dict(files=[IMAGE], masks={IMAGE: []}, counters={}))
    if validate is not None:
        stages["stage10_validate"] = lambda *a: validate
    out = io.StringIO()
    with mock.patch.multiple(pipeline, **stages), contextlib.redirect_stdout(out):
        result = pipeline.run(mode, "cpu", work=tmp)
    return result, out.getvalue()


class PipelineFailures(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.tmp = Path(d.name)

    def test_a_failed_validation_raises_in_both_modes(self):
        # The real stage 10 reads a CSV whose two instances overlap. The sha256 matches, so only validation can fail.
        for mode in ("full", "replay"):
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, "failed validation"):
                run_pipeline(self.tmp, mode, pipeline.SUBMITTED_SHA256, overlapping_csv())

    def test_b_replay_sha_mismatch_raises(self):
        with self.assertRaisesRegex(RuntimeError, "did not rebuild the submitted file"):
            run_pipeline(self.tmp, "replay", OTHER_SHA, validate=True)

    def test_b_replay_sha_match_passes(self):
        result, printed = run_pipeline(self.tmp, "replay", pipeline.SUBMITTED_SHA256, validate=True)
        self.assertTrue(result["matches_submitted"])
        self.assertIn("MATCHES", printed)

    def test_c_full_mode_sha_mismatch_is_reported_and_does_not_raise(self):
        result, printed = run_pipeline(self.tmp, "full", OTHER_SHA, validate=True)
        self.assertFalse(result["matches_submitted"])
        self.assertTrue(result["checks_passed"])
        self.assertIn("differs from the submitted", printed)


class VerifyCpuChainExitCode(unittest.TestCase):
    def run_tool(self, sha):
        """Run the tool's ``main`` with its stages replaced. Returns what it printed."""
        submitted = mock.MagicMock()
        submitted.make_submission.return_value = dict(instance_rows=0, sha256=sha)
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(verify_cpu_chain, "score", mock.MagicMock()), \
                mock.patch.object(verify_cpu_chain, "select", mock.MagicMock()), \
                mock.patch.object(verify_cpu_chain, "submit", submitted), \
                mock.patch.object(sys, "argv", ["verify_cpu_chain", "--rows", tmp, "--maps", tmp, "--out", tmp]), \
                contextlib.redirect_stdout(out):
            verify_cpu_chain.main()
        return out.getvalue()

    def test_differs_exits_non_zero(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_tool(OTHER_SHA)
        self.assertEqual(caught.exception.code, 1)

    def test_matches_returns_normally(self):
        self.assertIn("MATCHES", self.run_tool(verify_cpu_chain.SUBMITTED))


class SolverStatus(unittest.TestCase):
    # Two rows that conflict: the solver should keep the first one, which has the larger weight.
    W, EA, EB = np.array([2.0, 1.0]), np.array([0]), np.array([1])

    def test_d_non_optimal_status_raises(self):
        # Status 1 is a time or iteration limit. Its solution is usable, and the old check would have accepted it.
        for status in (1, 2, 3, 4):
            r = types.SimpleNamespace(status=status, x=np.array([1.0, 0.0]), message="synthetic")
            with self.subTest(status=status), mock.patch.object(select, "milp", return_value=r), \
                    self.assertRaisesRegex(RuntimeError, f"status {status}"):
                select._solve_component(self.W, self.EA, self.EB, None, None)

    def test_d_optimal_status_keeps_the_solution(self):
        keep = select._solve_component(self.W, self.EA, self.EB, None, None)
        self.assertEqual(keep.tolist(), [True, False])


class CustomShift(unittest.TestCase):
    # The stored mask covers rows 110-129 of columns 100-119. Rows 109 and 130 sit at 120, just under the cut of 128.
    LEVEL = 8                                   # The level is t = 0.50, so the cut is THR[8] = 128.
    COLS = range(100, 120)

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        P = np.zeros((S.H, S.W), np.uint8)
        P[108:132, 100:120] = np.array([40, 120] + [200] * 20 + [120, 40], np.uint8)[:, None]
        self.own = {0: P, 2: P, 4: P, 5: P, 6: P}
        stored = self.pixels(110, 129)
        t = S.LEV32[self.LEVEL]
        self.rows_file = Path(d.name) / "rows.npz"
        np.savez(self.rows_file, set_flat=stored, set_len=np.array([len(stored)]), fam=np.array([0]),
                 lvl=np.array([self.LEVEL]), F=np.array([[[t]]], np.float32))

    def pixels(self, first_row, last_row):
        return np.array([y * S.W + x for y in range(first_row, last_row + 1) for x in self.COLS], np.int64)

    def masks(self, *shift):
        one = np.array([0], np.int64)
        return submit.image_instances(self.rows_file, one, one, np.array([0.9]), self.own, *shift)

    def test_e_custom_shift_changes_the_mask(self):
        default = self.masks()[0]
        custom = self.masks((-1.0, 0.0))[0]
        # The default of -0.23 px lifts row 109 above the cut and keeps row 129.
        self.assertTrue(np.array_equal(default, self.pixels(109, 129)))
        # A full pixel moves row 130's value of 120 into row 129, so row 129 falls under the cut.
        self.assertTrue(np.array_equal(custom, self.pixels(109, 128)))
        self.assertFalse(np.array_equal(default, custom))

    def test_e_shift_none_keeps_the_stored_mask(self):
        self.assertTrue(np.array_equal(self.masks(None)[0], self.pixels(110, 129)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
