# Solar filament segmentation for filament-segmentation-2026

This repository has the code, the trained models and a demo notebook behind my entry (team BarrelAged) in the Kaggle
competition [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026). The task is
to find each solar filament in GONG H-alpha images of the whole Sun and outline it as its own mask. Entries are
scored by panoptic quality (PQ): a predicted mask only counts if it overlaps an annotator's drawing of the same
filament by more than half (IoU > 0.5).

My evaluated submission is `s1-S-t024.csv`, which scored 0.41 on the public leaderboard. The code here rebuilds it
from the test images.

## How it works

I started with the obvious approach: a U-Net that paints a probability map of "filament or not", then a threshold,
with each connected blob taken as one filament. That scored 0.383 on my development images. It breaks in a
predictable way. A single threshold either merges neighboring filaments into one blob or splits a long filament into
pieces, and the annotators often disagree about where a filament ends. Either way the masks miss the IoU > 0.5 bar.

So I stopped picking one threshold. The pipeline makes many candidate masks for each image (every blob at every
threshold level, plus merges of nearby blobs), a trained model predicts how well each candidate will match the
annotators, and the code then picks the best set of candidates that don't overlap. That alone took the development
score to 0.438. From there I gave the scorer more evidence about each candidate, namely what several other models
think of it, and refined how the final set is chosen. The final model scores 0.4706 on the development images.

Inference only reads the test H-alpha images. These are the steps, in the order the code runs them:

| # | What the step does | Module | Runs on |
|---|---|---|---|
| 1 | Reads the 180 test images. | `filseg/data.py` | CPU |
| 2 | Three U-Nets (ConvNeXt-T, a longer-trained ConvNeXt-T and ResNet34) each make a probability map. Each U-Net has 4 copies, one per training fold, and their maps are averaged. | `filseg/dense.py` | GPU, or CPU |
| 3 | A Mask R-CNN detector, also 4 fold copies, proposes filament masks of its own. | `filseg/det.py` | GPU, or CPU |
| 4 | A second detector (I6) finds filament regions and a small network refines their masks. | `filseg/i6.py` | GPU, or CPU |
| 5 | The code builds the candidate masks from the U-Net maps and describes each one with 86 numbers ("evidence"): its shape, its brightness, and how well it agrees with the other maps and detectors. [`docs/features.md`](docs/features.md) lists all 86. | `filseg/evidence.py`, `filseg/rows.py` | CPU |
| 6 | Ten LightGBM models and ten ExtraTrees models predict each candidate's expected IoU with the annotators. The score is 0.7 × the LightGBM average + 0.3 × the ExtraTrees average. | `filseg/score.py` | CPU |
| 7 | For each image, an exact integer program picks the set of non-overlapping candidates with the highest total score. Each candidate's score is first adjusted: s = q − 0.2·\|t − 0.475\| − 0.03 for candidates that aren't plain blobs of the main map, where q is the predicted IoU and t the candidate's threshold level. Only candidates with s above τ = 0.24 can be picked. | `filseg/select.py` | CPU |
| 8 | Each picked blob moves 0.23 pixels toward the top of the image, where the annotators' drawings sit on average (merged candidates stay as they are). Where two masks then touch, the higher-scoring one keeps the pixels. | `filseg/shift.py` | CPU |
| 9 | The code writes the masks to the submission CSV in the competition's format (COCO run-length encoding). | `filseg/write.py`, `filseg/submit.py` | CPU |
| 10 | The code checks the CSV and can score predictions with a copy of the competition metric. | `filseg/validate.py`, `filseg/metrics.py` | CPU |

The technical report explains why each step is there and what it adds, with numbers.

## Quick start

1. Use Python 3.10 and install the packages:
   `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128`
2. Download the competition data from Kaggle and unzip it. Set `FILSEG_DATA` to the folder that holds
   `test/test_images`. If you don't set it, the code looks in `data/MAGFiLO_1.0_Kaggle_2026` inside this repository.
3. Open `demo.ipynb` and run it from top to bottom. Intermediate files go to `FILSEG_WORK` (by default, `work/`).

You don't need to download anything else. The trained models are in `weights/`, and nothing is fetched from the
internet at inference time.

## Reproducing the submission

The notebook has two modes.

- **Replay mode (the default) rebuilds the submitted file exactly.** The `replay/` folder holds the outputs of steps
  2-4 from the run that produced my submission (98 MB). The notebook checks every file's hash, runs steps 5-10 on
  them and writes a CSV that is byte-for-byte identical to `submission/s1-S-t024.csv`. This needs only a CPU and
  takes about 20 minutes.
- **Full mode runs every step from the test images.** Steps 2-4 then run on your own GPU. Different GPUs do
  floating-point math slightly differently, and the U-Nets and I6 run in reduced precision (bfloat16), so you get a
  very close copy rather than the same bytes. When I ran full mode on an RTX 3070 Ti, 1,203 of the 1,206 submitted
  masks were matched at IoU > 0.5, and 910 of those were pixel-identical. Three masks were missing and five were new.
  The ResNet34 maps, which were originally made on that same GPU, came out bit-for-bit identical.
  `tools/compare_csv.py` compares your file with the submitted one.

`tools/verify_*.py` check each step against the stored outputs of the original run.

## Hardware and run time

Full mode took about 75 minutes on one RTX 3070 Ti (8 GB) with a 20-thread CPU:

| Step | Time |
|---|---|
| Reading the images | 14 s |
| U-Net maps (cnx, cnxL, R34) | 675 + 659 + 395 s |
| Mask R-CNN, 4 folds | 1,154 s |
| I6 | 656 s |
| Candidates and evidence (4 processes) | 754 s |
| Scoring and selection | 21 s |
| Shift and CSV writing | 152 s |

## Training

`training/` has the training scripts (`prep_train.py`, `train_dense.py`, `train_det.py`), and `configs/` has the
settings of every trained model. The notebook's training section is off by default, so the notebook loads the stored
models. All the networks were trained on development folds 0-3. The folds are fixed groups of observation days, so
images taken on the same day never end up in different folds. The technical report gives the split and every
parameter.

`training/scorer_data/` holds the two tables that the step-6 scorers were trained on (707 images for LightGBM, 566
for ExtraTrees; see its README). `filseg.train.fit_scorers()` retrains all 20 scorer models from them on a CPU in a
few minutes, and the retrained LightGBM files come out byte-identical to the shipped ones. The code that builds these
tables from the networks' out-of-fold maps isn't part of this release.

## What's in the repository

```
filseg/     the pipeline, one module per step
docs/       features.md: what each of the 86 candidate features measures
training/   training scripts; scorer_data/ holds the scorers' training tables
configs/    settings of every trained model, and the data split
weights/    the 38 trained model files, split into parts under 95 MB (filseg/weights.py joins them back)
replay/     the original run's step 2-4 outputs, for replay mode
tools/      checking and packaging scripts
submission/ the submitted s1-S-t024.csv, for comparison
demo.ipynb  the end-to-end demo
```

## License and data

- The code is released under the Apache License 2.0 (see `LICENSE`).
- I used the competition data under the competition's terms, which allow non-commercial and academic use. This
  repository doesn't include the images or annotations. The trained models, `replay/` and `training/scorer_data/`
  are derived from them.
- The third-party libraries keep their own licenses (see `requirements.txt`).

## Citations

The data come from the MAGFiLO dataset and the competition built on it:

1. Ahmadzadeh, A., et al. (2024). A dataset of manually annotated filaments from H-alpha observations. *Nature
   Scientific Data* 11, 1031. https://doi.org/10.1038/s41597-024-03876-y
2. Ahmadzadeh, A., Kempton, D. J., Li, Q., and Pevtsov, A. A. (2026). *Solar Filament Segmentation Challenge 2026*.
   Kaggle. https://kaggle.com/competitions/filament-segmentation-2026

This work uses GONG data obtained by the NSO Integrated Synoptic Program, managed by the National Solar Observatory,
which is operated by the Association of Universities for Research in Astronomy (AURA), Inc. under a cooperative
agreement with the National Science Foundation and with contribution from the National Oceanic and Atmospheric
Administration. The GONG network of instruments is hosted by the Big Bear Solar Observatory, High Altitude
Observatory, Learmonth Solar Observatory, Udaipur Solar Observatory, Instituto de Astrofísica de Canarias, and Cerro
Tololo Interamerican Observatory.

## How I used AI tools

I built this project with AI coding agents, which the competition rules allow (Rules §2.8.a.1(b)). I wish to be transparent that my use of AI agents may be more generous than the judges would prefer.

- **What LLMs did.** Wrote the code, ran the experiments, kept the experiment records
  and organized this release. Separate agents checked the work against those records. 
- **What I did.** I set the direction and approved the compute and data the work used. I chose which methods to
  adopt (for example, the B3f selection rule, even though its pre-registered main test had failed), chose the
  Kaggle submissions, and made the judgment calls, such as keeping the 0.23 px shift. 
- **How the results were checked.** Each step of this release was compared with the stored outputs of the run that
  produced my submission (`tools/verify_*.py`), and replay mode rebuilds the submitted file byte for byte.
