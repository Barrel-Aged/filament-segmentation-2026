# I6 results: interim learned ROI production benchmark

I6 is complete. Kaggle submission **57712851** is **COMPLETE**, with returned public score **0.37**. The displayed public score improves on H8/S1 (0.30). This is a public competition benchmark, not another development fold score, an exact estimate of the private score, or evidence that all recall/morphology tradeoffs have disappeared.

## Frozen production recipe

One detector/refiner pair was trained from generic COCO/ImageNet initialization on all 707 supplied physical training images and 1,154 separate annotator records. Seed 20261014 and fixed 15k/30k schedules transferred from I5. Two additional half-fit pairs, at 7.5k/15k, produced candidates for the opposite complete UTC-day groups. Their union covers all 707 images without training-group leakage into selector targets. The production selector fitted **122,215** pairwise examples from those predictions.

Mask threshold .5 and confidence .8 were frozen by a prospective consensus rule over I5 inner selections, with area 64, cluster limit 6 and margin .02. The .4/.5 mask vote tied; the predeclared nearest-.5 tie rule selected .5. This is a deployment transfer convention, not new full-data validation. No calibration, checkpoint choice, manual mask edit, ensemble or resubmission followed test appearance or the public result.

All six fits completed **90,000 successful updates and 600,000 sample occurrences**. Full-data training uses the same update budget as development, so per-image exposure differs. The old fold4 enters only authorized production fitting; no new holdout score was computed. Production weights remain excluded from research initialization and evaluation.

| Fit | Images | Updates | Samples | Training minutes | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|
| I6-detector-full-s20261014 | 707 | 15,000 | 60,000 | 89.77 | 2178.5 |
| I6-refiner-full-s20261014 | 707 | 30,000 | 240,000 | 46.96 | 880.6 |
| I6-detector-full-s20261014-m0 | 352 | 7,500 | 30,000 | 45.03 | 2144.9 |
| I6-refiner-full-s20261014-m0 | 352 | 15,000 | 120,000 | 23.52 | 880.6 |
| I6-detector-full-s20261014-m1 | 355 | 7,500 | 30,000 | 44.99 | 2178.5 |
| I6-refiner-full-s20261014-m1 | 355 | 15,000 | 120,000 | 23.51 | 880.6 |

## Test export and Kaggle receipt

The learned export contains **1,348 instances** across 177 of 180 inferred images, with 3 explicit empty-image records retained in the prediction evidence. The matched confidence-only control contains **1,426 instances** across 177 images; it was exported and audited but not submitted.

Every test image was inferred twice in separate guarded Linux GPU processes. Raw detector/refiner digests, candidate records and decoded instances matched exactly. Independent Linux and Windows CPU reconstruction verified both CSVs, full 2048 dimensions, unique sequential IDs, disjoint instances, COCO column-major orientation and exact mask round trips. The original confidence assembler independently reproduced the confidence control. Three prospectively selected raw/outline panels were actually viewed for orientation/export sanity only; no test labels were available and no settings changed.

Submitted CSV SHA-256: `e4a7b43e20f6c3ddf2552abe28194e77c30122122f7d5dbac20fa06fbe739072`. One token-only CLI upload through `scripts/kaggle.ps1`, with a pre-upload attempt marker and unique description. Submitted UTC: 2026-10-27 20:48:04.187000. [Competition submissions](https://www.kaggle.com/competitions/filament-segmentation-2026/submissions). The confidence control remains unsubmitted. Public and development scores use different data; the prior I5 all-eight development PQ is 0.419485, confidence 0.406366, H7 0.369587.

## Costs and verification

Counted GPU-stage time: **5.425 hours**, entirely on guarded BarrelAgedWorkflows. Counted CPU-stage/transfer wall time: **1.100 hours**, under the six-hour bound. Retained Linux artifacts including the full archive: **5.664 GiB**; Windows artifacts: **5.664 GiB**, within 20/25 GiB. All originals remain. Every retrieved archive member was SHA-256 verified.

Learned-selector cold first image: 5.27s; mean 2.214s/image; p95 2.373s; peak allocated GPU memory 699.8MiB. Confidence mean 2.211s/image. These retain I5's convention: mean includes the first image, selector cold includes preceding confidence decoding, startup excludes OS process creation, caches were not flushed, and allocated memory is not total/reserved VRAM. No new performance threshold was selected after seeing these measurements.

All schedules/sample identities, six checkpoints, generic initialization digests, 192 sampled input/target tensors across hosts, group exclusions, prospective source/Git freezes, two historical GPU parity images and the weighted ridge solve passed. Ridge was independently solved from the preserved matrix; this does not independently regenerate every pair target. Target generation is bound to frozen source, supplied annotations and group-excluded candidate hashes. Twenty CPU regressions passed on each host. Linux owned-child RSS was checked every five seconds under 24 GiB; this is not a continuous peak guarantee. Accounting excludes uninstrumented interactive preparation/review/submission and final metadata packaging. Windows used no GPU.

## Interpretation and next work

The displayed public score improves on H8/S1 (0.30). The submission benchmarks I5's confirmed PQ-focused recipe. **H7 remains the balanced reference**: I5 learned selection improved faint recall from 24.8% to 30.4%, but ordinary recall fell 73.0% to 66.7%, small 47.2% to 36.4%, and limb 61.0% to 50.5%; merge flags also increased. A public PQ result cannot establish recovery of those costs or qualitative ranking.

The next improvement phase should test an object-level H7/ROI combination locally under a fresh protocol. First measure independently useful recovery and false-positive/merge costs, then compare each standalone model, a fixed simple combination and at most one training-group-excluded learned selector. Preserve reliable H7 ordinary/small/limb instances while testing whether ROI candidates add faint detections without replacing good objects. Do not reuse the rejected equal-probability blends automatically, fit on outer/test predictions, or use an oracle as a deployable selector.

If this public benchmark fails to transfer the local advantage, add a grouped development check of exposure/decoder/source-year transfer before more production fitting; do not tune against the leaderboard. Confirmation must cover four development folds/two seeds, perturbations, source/year, extent/connectivity, and complete runtime with unchanged recall/merge safeguards. These folds have been repeatedly observed; report that limitation rather than claiming a fresh holdout. Final delivery M10 remains deferred, with the November 13 buffer preserved.

Evidence: `outputs/improvement-i6/contract-audit.json`, `windows-audit.json`, `cost-audit.json`, `visual-review.json`, `kaggle-result.json`, and the full archive inventory. Acceptance is `manifests/improvement-i6.json`; the final delivery receipt records the clean Git commit and verified Linux backup.
