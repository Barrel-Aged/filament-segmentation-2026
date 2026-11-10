# Training scripts as they ran

This folder holds the original scripts behind the two training steps that `filseg/train.py` does not cover: training the
I6 detector and refiner, and building the labeled tables that the scorers train on. These are the scripts I actually used
during the competition, copied byte for byte. They are not a cleaned-up port.

**The notebook and inference don't depend on this folder.** Nothing in `demo.ipynb` or `filseg/` imports or runs anything
here. A judge doesn't need to run these scripts to reproduce the predictions.

**I have not re-tested these scripts in this location.** They ran from my own project folders on my own machines. Claude
copied them here on 2026-11-09 and checked that every copy is byte-identical to its original (`MANIFEST.tsv`), with
one exception: in the six `i6/runs/*/run.json` records I removed the path of an unrelated program from the list of
other programs that were using the GPU, and `MANIFEST.tsv` holds the hashes of the edited files. Claude did
not run any of them for this release.

**The scripts contain paths and machine checks from my setup, and they stay as they were.** Expect these things:

- Absolute paths such as `D:/FilamentSegmentationKaggle/claude` and `/home/barrelaged/projects/filament-claude`.
- Host guards (`filament/compute.py`, `cfil/guard.py`) that stop a GPU script unless it runs on my Linux box, whose user
  name and host name are in `configs/compute.json`. They read those values, and they contain no passwords or keys.
- `ssh` and `scp` calls to my Linux box, and virtual-environment paths like `.venv/bin/python`.
- Hash checks that refuse to run if a frozen file has changed, and guards that refuse to overwrite earlier outputs.

To run these yourself you would have to change the paths and remove the host checks, and the outputs would land in a fresh
workspace. Weights, images, annotations, logs and the large intermediate files are not in this folder.

Codex, an AI coding agent I used earlier in the project, built the I6 detector and refiner training code in `i6/`. Claude
wrote the scorer-table code in `scorer_tables/`.

## What's in here

| Folder | What it holds | Files |
|---|---|---|
| `i6/` | I6 data split, training, the half-fit "miner" models and the out-of-fold candidate banks, with the configs, data lists and run records | 52 |
| `scorer_tables/` | Every step from the fold models' out-of-fold outputs to `lgbm_707.npz` and `et_566.npz` | 47 |
| `MANIFEST.tsv` | Each file's sha256, its path here and its original path | |

The folder also has a `.gitattributes` that keeps line endings unchanged, because 12 of the copies use Windows line endings
and the release's own settings would otherwise rewrite them. Its `.gitignore` keeps the folders named `work/` tracked.

Five kinds of file are left out on purpose:

- **Steps that don't feed these tables.** `i6_selector.py`, `i6_export.py` and their helpers built Codex's own I6
  submission, and my final pipeline doesn't use them. `i6_inference_campaign.py` still lists them as the later steps of
  its chain. Lines that call experiment scripts such as `work/exp_feat.py` and `src/cfil/avg_maps.py` are the same kind of
  thing, and those files aren't here either.
- **Files the scripts check for before they start.** The two earlier run records
  `runs/G4-{detector,refiner}-f0-s20261014/run.json` (the I6 training script compares its starting model with them),
  four test files that the launcher runs first, and the I5-era files the inference freeze checks
  (`scripts/i5_candidates_gpu.py`, `configs/enhancement-h2.json`, `manifests/improvement-i5-splits.json` and files
  under `outputs/growth-6/`). You would have to remove those checks to run the scripts.
- **The COCO starting weights for the detector.** The script expects them at
  `checkpoints/pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`, which is torchvision's public file.
- **Later checks.** `train707.py repro566` needs `score_select.py` from the test stage, and it is only a check.
- **Codex's I5 result files and the wider source tree.** `prepare_i6.py` reads I5's result files for a threshold vote, and
  the vote's outcome is recorded in `i6/configs/improvement-i6.json`. Codex's `verify()` also re-hashes every file in the
  source freeze (`i6/outputs/improvement-i6/*-source-freeze.json`), which covers all of Codex's `src/` tree and is more
  than the files needed here.

## Are these the versions that ran?

Mostly yes, and the exceptions are listed here.

- **I6.** Every script, source file, config and data list here that was frozen before the runs still matches its frozen
  hash. `i6_ops.py` and `i6_inference_ops.py` were not in the freezes, so this check doesn't cover those two.
- **`scorer_tables/`.** Every file matches its last commit in my private git history for this part of the project, except
  `export_tables.py`, which was never committed. The manifests stored with the runs
  (`work/ideas8/bundle-hold/manifest.json` and `work/ideas8/b3f707/manifest.json`, both copied here) record the hashes of
  the row-building code, and every file they list that is in this folder still matches: `build_rows.py`, `holdrows.py`,
  `xcomb.py`, `xcands.py`, `xcands2.py`, `pool_oracle.py`, `xcv.py`, `xm_rows.py`, `run_arms.py`, `cands.py`,
  `build_cands.py`, `rich.py`, `xmap.py`, `stack.py` and `train707.py`.
- **`src/cfil/det.py` changed after the detector runs.** The copy here is the current file, with later additions for
  local runs (resume, CPU device, pause and stop controls). The detector's out-of-fold and fold-4 runs on 2026-10-29 used
  earlier commits. `det.py.commit-3516743` is the file from the commit nearest those runs (18:06 MDT that day). The file
  changed between the runs, so I can't say which commit each root used.
- **`src/cfil/dense.py`** gained one line (`CFIL_ROOT`) after the cnx, cnxL and R34 out-of-fold maps were made.
- **`src/cfil/build_cands.py`** stores two extra arrays that the earliest development candidate files don't have. Those
  files were made before that change, and `xcomb.py` later rebuilt the same candidates from scratch on all 566 images.

## The order the steps ran

"Recorded" means the command is written in a script or log. "Reconstructed" means no record of the exact command survives
and I worked it out from the script's arguments.

### Part 1: I6 (Codex, 2026-10-27)

| Step | Command | Status |
|---|---|---|
| 1. Fix the data split, thresholds and protocol | `python scripts/prepare_i6.py` | Reconstructed |
| 2. Freeze the training sources | `python scripts/freeze_i6_training.py` | Reconstructed |
| 3. Send the sources to the GPU box and run checks | `python scripts/i6_ops.py deliver`, which runs `i6_ops.py preflight` there | Reconstructed |
| 4. Launch training | `python scripts/i6_ops.py launch`, which starts `.venv/bin/python scripts/i6_training_campaign.py` | Reconstructed launch, recorded chain |
| 5. Six fits, in order | `i6_train_gpu.py train --cell full-s20261014 --stage detector\|refiner [--miner 0\|1]` | Recorded in `i6_training_campaign.py` |
| 6. Audit each fit | `audit_i6_fit.py checkpoint\|fit` with the same options | Recorded in `i6_training_campaign.py` |
| 7. Freeze the inference sources | `python scripts/freeze_i6_inference.py` | Reconstructed |
| 8. Launch the out-of-fold banks | `python scripts/i6_inference_ops.py deliver`, then `launch`, which starts `i6_remote_chain.py` and `i6_inference_campaign.py` | Reconstructed launch, recorded chain |
| 9. Candidates for the opposite halves | `i6_candidates_gpu.py oof --miner 0`, then `--miner 1` | Recorded in `i6_inference_campaign.py` |

No runbook lists the I6 launch commands, so the order above comes from reading the scripts. The two "miner" pairs train on
separate halves of the 707 images, 352 and 355 images, split by UTC day group. Each one then proposes candidates for the
half it didn't see. The halves span all five folds, so each miner trained on some fold-4 images, and an image's features
always come from the miner that didn't train on it. The full-fit pair trains on all 707 images and is the one that runs on
the test set. The seed is 20261014.

| Fit | Images | Updates | Training seconds | Weights sha256 |
|---|---:|---:|---:|---|
| Detector, full | 707 | 15,000 | 5,386 | `dd5e568bbbff8afaf7d52c2f60071f2b708529e04d13b28ea4596e8e59e0588d` |
| Refiner, full | 707 | 30,000 | 2,817 | `c417acf9200f7c4d53fff1fb08270a2a3748e7720212f5282fbb210e33702d01` |
| Detector, miner 0 | 352 | 7,500 | 2,702 | `b638425424e1b95ad4ad16922f3a39124384d342ae9aadf236e945cfbcc1b565` |
| Refiner, miner 0 | 352 | 15,000 | 1,411 | `c7336e3f60bbf6f533ac06c4a59fe01fb5dbefc9799ed7f6ec86d1dc8ade7e71` |
| Detector, miner 1 | 355 | 7,500 | 2,699 | `9f0e732e8d9e58698742a79d849bd0cf4edd40cea80919e5c35b83eaba0bc49e` |
| Refiner, miner 1 | 355 | 15,000 | 1,410 | `c77b3ffaa6a28a9464842d3f5416c8cfe1fb2376beebbe70e31de9ebb4a5d2e9` |

- **Hardware:** one RTX 4070 SUPER (12,282 MiB) on my Linux box. The six fits took 17,157 GPU-stage seconds, and their
  audits added 716 CPU-stage seconds (`i6/outputs/improvement-i6/training-execution.json`).
- **Out-of-fold banks:** `oof-m0` took 771 seconds and `oof-m1` took 751 seconds on the same GPU
  (`inference-execution.json`).
- **Inputs:** the 707 training images, the competition annotations, `manifests/dataset-v1.json`,
  `manifests/split-v2.json`, `configs/growth-4.json` (the fit recipes), and the COCO and ImageNet starting weights.
- **Outputs:** `runs/I6-*/weights.pt` (hashed above) and `outputs/improvement-i6/candidates/oof-m0` and `oof-m1`, one JSON
  file per image plus a manifest. The manifest files aren't stored with their hashes anywhere else. Their sha256 values,
  computed on 2026-11-09, are `7f0f85d7995eb49f5b458d51f6924707d6b6b0fdac850411ba3d5b02ef66c902` and
  `3dd3a49075c64d139da228e94837e0a72fcaeb7c4fbec548c289d393266817a1`.
- **Environment:** `i6/uv.lock` and `i6/training/uv.lock`. Each fit's run record is `i6/runs/I6-*/run.json`.

### Part 2: The scorer tables (Claude, 2026-10-28 to 2026-11-06)

Times are as recorded. "Unrecorded" means no record exists. GPU work ran on the remote RTX 4070 SUPER, except where noted.
CPU work ran on my Windows PC, with Python 3.10.6, numpy 2.2.6, scipy 1.15.3, scikit-image 0.25.2, OpenCV 5.0.0 and
LightGBM 4.7.0. The GPU library versions at the time of these runs weren't recorded.

| # | Step | Command | Status | Time and hardware |
|---|---|---|---|---|
| 1 | Build the image cache | `.venv/bin/python claude/src/cfil/prep.py /home/barrelaged/projects/filament-claude` | Reconstructed | Remote CPU, unrecorded |
| 2 | Train the 12 U-Net fold models and write their out-of-fold maps | cnx: `queue_dense.sh cnx unet tu-convnext_tiny 24000 --workers 5` (reconstructed). cnxL: `start_cnxL.sh`, `resume_cnxL3.sh` (recorded). R34: `queue_dense.sh r34 unet resnet34 30000` (reconstructed). | Mixed | Remote RTX 4070 SUPER. Map inference per fold: cnx 187, 188, 187 and 95 s. cnxL 157, 157, 104 and 94 s. R34 116, 116, 115 and 116 s. Training times are in `filseg/train.py`. |
| 3 | Train the 4 Mask R-CNN fold models and write their out-of-fold candidates | `queue_det.sh det "0"` (fold 0, reconstructed), then `after_det_f0.sh`, which runs `queue_det.sh det "1 2 3"` (recorded) | Mixed | Remote RTX 4070 SUPER. Inference per fold: 212, 148, 200 and 211 s. |
| 4 | Fold-4 maps | cnx and cnxL: `queue_final2.sh`, `resume_cnxL3.sh` (recorded). R34: `CFIL_LOCAL_GPU=1 CFIL_ROOT=... dense.py infer --models r34-f0,...,r34-f3 --split 4 --out work/maps/final/holdr34.npy` (reconstructed) | Mixed | cnx 755 s and cnxL 364 s on the remote GPU. R34 365 s on my PC's RTX 3070 Ti. |
| 5 | Fold-4 detector candidates | `queue_final2.sh`, per fold model | Recorded | Remote GPU, 211, 151, 150 and 151 s |
| 6 | Copy to the Windows PC | `fetch_maps.sh`, `fetch_det.sh` | Recorded in scripts | Unrecorded |
| 7 | Candidate tables on the cnx maps | `build_cands.py`, called from `work/run_4fold.sh` (dev) and `work/run_final_feats.sh` (fold 4) | Recorded in scripts | Windows CPU, unrecorded |
| 8 | Dev I6 and detector evidence | `work/run_i6oof.sh` (`ext.py --greedy-sel`, `rich.py`), `work/run_det_eval.sh 0` (reconstructed) and `work/run_det_4fold.sh` (folds 1-3, recorded) | Mixed | Windows CPU, unrecorded |
| 9 | Dev second-opinion checks | `work/run_cnxL_4fold.sh` and `work/run_xmR.sh` (`xmap.py`) | Recorded in scripts | Windows CPU, unrecorded |
| 10 | Fold-4 evidence | `work/run_c2_inputs.sh`, which runs `run_final_feats.sh cnx`, and `work/run_c2plus.sh` for `xmap.py` | Recorded in scripts | Windows CPU, unrecorded |
| 11 | Dev rows | `python work/ideas7/playbook/xcomb.py --nproc=5` | Reconstructed | Windows CPU, 992 s |
| 12 | Dev second-opinion columns | `python work/ideas7/exp/p1_poolxm/xm_rows.py --nproc=5`, in two batches for folds 0-1 and 2-3 | Reconstructed | Windows CPU, 722 s and 649 s |
| 13 | Fold-4 rows | `python work/ideas8/bundle-hold/build_rows.py --nproc=5` | Reconstructed | Windows CPU, 419 s |
| 14 | Column and row checks | `python -B work/ideas8/b3f707/train707.py cols` | Reconstructed | Windows CPU, 3 s |
| 15 | Fit the 10 LightGBM models | `python -B work/ideas8/b3f707/train707.py fit` | Reconstructed | Windows CPU, 93 s |
| 16 | Export the two tables | `python work/package/release_work/scorer_train/export_tables.py` | Reconstructed | Windows CPU, 19 s |

Step 15 fits the LightGBM models in `weights/models/`. The ExtraTrees models were fit by a separate short script that is
not copied here, and `filseg.train.fit_et` is the same recipe. `filseg/train.py` can repeat both from the exported tables. Step 16 is what
wrote the files in `training/scorer_data/`.

### Inputs and outputs of Part 2

- **Inputs:** the 707 training images and the competition annotations, `configs/split-v2.json`, and the I6 banks from
  Part 1. Fold 4 uses the same two I6 banks as the development folds.
- **Targets and labels:** `cfil/cands.py` measures each candidate's IoU with every annotator's filaments, and `xcomb.py`
  and `build_rows.py` store those per-record IoUs as `hits`. `train707.py` turns them into the target `y = hits.mean(1)`
  and the weight `w`, the number of records, a quarter of that for fold-4 rows.
- **Dev evidence** comes from fold models that didn't train on the image, and from the I6 miner that didn't train on it.
  **Fold-4 evidence** comes from the 4-fold averages and all four detector copies.

## Recorded hashes of the outputs

| Output | Hash | Where recorded |
|---|---|---|
| Remote cache `meta.json` | `8b88835ebc03b246851d86abef74c39c0cb0b2cf61c520f0df6d410d5a52f52f` | A later provenance check (`v1_remote.log`) |
| Fold-4 map `holdcnx.npy` | `c1d817d794f2b2e37e82f0cefef6a037616618f59fb38c1f316c6cf9deaa75c0` | `bundle-hold/manifest.json` |
| Fold-4 map `holdcnxL.npy` | `dc88a93374a132a81fcfcc92210cf2c136fcf34347f3036a09b7e2b09f17b930` | `bundle-hold/manifest.json` |
| Fold-4 map `holdr34.npy` | `127ea0d08653e812fe4bc9471815d527d695707aab16f6540d11f8a0fa91fcc0` | `bundle-hold/manifest.json` |
| Fold-4 rows (141 files, directory digest) | `b9713fba9e754b1046eee003a84c3c52cc6df3dc8c788f975d96cb7910e4594b` | `b3f707/manifest.json` |
| Dev rows `xcomb` (566 files, directory digest) | `f65a7fb265f2758e35269fdc0b1293bab7e37bd9509132fe119e9e275d9f592a` | `b3f707/manifest.json` |
| Dev columns `xm` (566 files, directory digest) | `c078e2ece100ee20ad503ae50e520d5ad988ddc77d49cada3e1bbe1454bd224f` | `b3f707/manifest.json` |
| `sha256(X, y, w)` of the 707-image table | `cea083c3e5d7639393dd3806a1615c3c8084dec0e1e3752bab7cc2ad0a9e5b2a` | `train_meta.json` (the models folder) |
| `sha256(X, y, w)` of the 566-image table | `8cc727fba431efe9dc5fe7409c9ad2393795fa1018423fe613d2e8206ad04f8d` | `train_meta.json` |
| `training/scorer_data/lgbm_707.npz` | `5c4a5865367bb5b69974621a1f80185765a1e055acef14f955f60abd7b2e9859` | `export_tables.log` (matches the release file) |
| `training/scorer_data/et_566.npz` | `d73b20177a3a8e35cd1c4ba66b41446a7906cf80a552dbfdd994090c5a3e3eef` | `export_tables.log` (matches the release file) |

Hashes for the out-of-fold maps, the fold detector roots, the dev evidence folders and the per-image I6 candidate files
aren't listed here. Of those, the out-of-fold maps have no recorded file hash, and the fold-4 evidence folders have
directory digests inside `bundle-hold/manifest.json`.
