# Solar filament instance segmentation: filament-segmentation-2026

Code, trained weights and a demo notebook for team BarrelAged's entry in the Kaggle competition
[filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026): instance segmentation
of solar filaments in GONG H-alpha full-disk images, scored by panoptic quality (PQ) pooled over annotators.

The evaluated submission is `s1-S-t024.csv` (Kaggle public score 0.41). This repository rebuilds it from the test
images.

## The pipeline

Stages 2-4 propose candidate filament masks, stages 5-6 score every candidate, and stages 7-9 pick a conflict-free
set and write it. Inference reads only the test H-alpha images.

| # | stage | module | device |
|---|---|---|---|
| 1 | read the 180 test images | `filseg/data.py` | CPU |
| 2 | dense probability maps: three U-Nets (ConvNeXt-T, a longer-trained ConvNeXt-T, ResNet34), 4 fold models each, averaged | `filseg/dense.py` | GPU (CPU possible) |
| 3 | detector candidates: Mask R-CNN, 4 fold models | `filseg/det.py` | GPU (CPU possible) |
| 4 | learned-ROI detector + refiner candidates (Codex I6; see Credits) | `filseg/i6.py` | GPU (CPU possible) |
| 5 | candidate masks (threshold component trees plus merges) and 86 evidence columns per candidate | `filseg/evidence.py`, `filseg/rows.py` | CPU |
| 6 | expected IoU of each candidate: q = 0.7 × mean of 10 LightGBM + 0.3 × mean of 10 ExtraTrees | `filseg/score.py` | CPU |
| 7 | exact selection: per image, the non-overlapping set maximising Σ(s − τ), s = q − 0.2·\|t − 0.475\| − 0.03·[not a cnx tree node], τ = 0.24 (integer program, HiGHS) | `filseg/select.py` | CPU |
| 8 | 0.23 px sub-pixel shift toward row 0 of tree and U-Net masks, then pixel claiming in descending s | `filseg/shift.py` | CPU |
| 9 | column-major COCO RLE CSV | `filseg/write.py`, `filseg/submit.py` | CPU |
| 10 | submission checks and the PQ metric | `filseg/validate.py`, `filseg/metrics.py` | CPU |

## Quick start

1. Python 3.10. `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128`
2. Download the competition data from Kaggle and unzip it. Point `FILSEG_DATA` at the folder that holds
   `test/test_images` (default: `data/MAGFiLO_1.0_Kaggle_2026` inside this repository).
3. Open `demo.ipynb` and run it top to bottom. Intermediate files go to `FILSEG_WORK` (default `work/`).

No other download is needed: the weights are in `weights/`, and no pretrained backbone is fetched at inference.

## Reproducibility

- **Replay mode (exact).** `replay/` holds the stage 2-4 outputs of the run that produced the submission (98 MB,
  every file sha256-checked on unpacking). With replay on, the notebook runs stages 5-10 on them and writes a file
  byte-identical to the submitted `s1-S-t024.csv` (sha256 `18bf87d6…74cd4`), on CPU, in about 20 minutes (4 processes).
- **Full mode (from the images).** Stages 2-4 run on your GPU. GPUs differ slightly in floating-point arithmetic and
  the U-Nets and I6 run in bfloat16, so the result is a close copy rather than the same bytes. On an RTX 3070 Ti,
  1,203 of the 1,206 submitted instances were matched at IoU > 0.5, 910 of them pixel-identical; 3 instances were
  missing and 5 new. The ResNet34 maps, originally computed on that same GPU, reproduced bit for bit.
  `tools/compare_csv.py` compares your output with the submitted file.
- **Stage checks.** `tools/verify_*.py` compare each stage with the original run's stored outputs.

## Hardware and runtime

Full mode on one RTX 3070 Ti (8 GB) and a 20-thread CPU took about 75 minutes:

| stage | time |
|---|---|
| images | 14 s |
| U-Net maps (cnx, cnxL, R34) | 675 + 659 + 395 s |
| Mask R-CNN, 4 folds | 1,154 s |
| I6 | 656 s |
| evidence and candidate rows (4 processes) | 754 s |
| scoring and selection | 21 s |
| shift and CSV writing | 152 s |

## Training

`training/` holds the training entry points (`prep_train.py`, `train_dense.py`, `train_det.py`), and `configs/` holds
the arguments of every trained run. The notebook's training section is optional and off by default; it loads the
stored weights. The networks were trained on dev folds 0-3 (the folds are fixed groups of UTC observation days); see
the technical report for the split, the scorer's training data and every parameter.

`training/scorer_data/` holds the two tables the stage-6 scorer was fit on (LightGBM: 707 images, ExtraTrees: 566;
see its README). `filseg.train.fit_scorers()` refits the 10 + 10 models from them on the CPU in a few minutes; the
refit LightGBM files are byte-identical to the shipped ones. The code that builds these tables from the out-of-fold
maps is not part of this release.

## Repository layout

```
filseg/     the pipeline, one module per stage
training/   training entry points; scorer_data/ holds the scorer's training tables
configs/    run arguments and the data split
weights/    38 model files split into parts under 95 MB (weights/manifest.json; filseg/weights.py rejoins them)
replay/     the original run's stage 2-4 outputs, for replay mode
tools/      verification and packaging scripts
submission/ the submitted s1-S-t024.csv, for comparison
demo.ipynb  the end-to-end demo (one call per stage to filseg/pipeline.py)
```

## Licences and data

- The code in this repository is released under the Apache License 2.0 (`LICENSE`).
- The competition data are used under the competition's data terms, for non-commercial and academic use. This
  repository does not redistribute the images or annotations. `replay/`, the trained weights and
  `training/scorer_data/` are derived from them.
- Third-party libraries keep their own licences (see `requirements.txt`).

## Credits

- Stage 4 (`filseg/i6.py`) is a trimmed copy of the I6 model built in this project's earlier Codex track; the module
  docstring lists the files it came from.
- Data: the MAGFiLO dataset and the GONG network of the U.S. National Solar Observatory.

<!-- OWNER TO DO: the exact citations from the report template. -->

## How this project used AI tools

This project was built with AI coding agents, as the competition rules allow (Rules §2.8.a.1(b)).
- **What the agents did.**
  - OpenAI Codex (2026-10-14 to 11-01) built the first track: the data split, the metric replica, the early models
    and the I6 detector used in stage 4.
  - Claude, through Claude Code (from 2026-10-29), built the dense U-Nets, the detector, the candidate scorer, the
    selection rule and this release.
  - In both tracks the agents wrote the code, ran the experiments, kept the experiment records, and separate
    agents checked the work against those records.
- **What the team decided.** The team member approved the compute and data use, made the adoption decisions put to
  them (for example, adopting the B3f decision rule after its pre-registered primary arm failed), chose
  every Kaggle submission and the final, ruled on the judgement calls (for example, keeping the 0.23 px shift), and
  chose how this release is reproduced and checked.
- **How the results were checked.** Each stage of this release was compared with the stored outputs of the run that
  produced the submission (`tools/verify_*.py`), and replay mode rebuilds the submitted file byte for byte.

The technical report describes every component, why it is there and what it contributes.
