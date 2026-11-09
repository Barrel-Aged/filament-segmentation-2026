# The 86 features of each candidate

Step 6 of the pipeline predicts, for every candidate mask, how well it will match the annotators (its expected IoU,
q). The scorers see each candidate as a row of 86 numbers. This page lists all 86 and says what each one measures.

I designed every one of these features (Claude wrote the code). None of them is learned. Many of them, though, are
computed from the outputs of trained models: the U-Net probability maps, Mask R-CNN and the I6 detector. The learning
happens in those models and in the step-6 scorers, which decide how much each feature matters.

The code is in `filseg/evidence.py` (columns 0-61) and `filseg/rows.py` (columns 62-85). The names below are the ones
stored in `training/scorer_data/*.npz` (the `columns` field).

## A few terms

- **The maps.** Three U-Nets each give a probability map, P, where every pixel holds the chance it's part of a
  filament. Each U-Net also gives a "spine" map for the filament's center line. *cnx* is the main map. *cnxL* and
  *R34* are the two other U-Nets.
- **Candidates.** The code thresholds the map at 16 levels (t = 0.10, 0.15, ..., 0.85) and takes every connected blob
  of at least 40 pixels at every level. Blobs at neighboring levels nest inside each other, like a tree: a blob at
  t = 0.50 sits inside a bigger blob at t = 0.45. On top of the plain cnx blobs, the pool also has blobs of the cnxL
  and R34 maps, and two kinds of merge that join nearby blobs.
- **Instances.** Mask R-CNN and I6 each output their own filament masks, called instances here, each with a
  confidence score.
- **IoU** is the overlapping area of two masks divided by their combined area.

## How much each group matters

Share of the deployed LightGBM scorer's split gain, by group: Mask R-CNN agreement 0.63, I6 agreement 0.20, the
candidate's own shape and map features 0.13, second opinions 0.04, family and "improvement" columns about 0. Gain
share says how often the scorer leans on a group, not how much PQ the group adds. The technical report gives the
measured PQ gains.

## Columns 0-27: the candidate itself

Computed on the candidate's own map. For cnxL and R34 blobs that's their own map; merges use the cnx map.

| # | Name | What it measures |
|---|---|---|
| 0 | t | The threshold level the blob was cut at. |
| 1 | lvl | The same level as an index, 0-15. |
| 2 | log_area | The natural log of the blob's pixel count. |
| 3 | bw | Width of the blob's bounding box. |
| 4 | bh | Height of the bounding box. |
| 5 | aspect | The box's long side divided by its short side. |
| 6 | fill | The blob's area divided by the box's area. |
| 7 | p_mean | Average map probability inside the blob. |
| 8 | p_max | Highest map probability inside the blob. |
| 9 | p_std | Spread (standard deviation) of the probability inside the blob. |
| 10 | p_edge | Average probability on the blob's 1-pixel inner boundary. |
| 11 | persist | p_max − t: how far the blob's peak sits above the threshold it was cut at. |
| 12 | par_ratio | The blob's area divided by its parent's area, its parent being the blob that contains it one level down (1 if it has no parent). |
| 13 | n_child | How many blobs it splits into one level up. |
| 14 | depth_up | How many ancestors it has in the tree. |
| 15 | s_mean | Average spine probability inside the blob. |
| 16 | s_max | Highest spine probability inside the blob. |
| 17 | contrast | Average image brightness in a ring up to 6 pixels wide around the blob, minus the average inside it. Filaments are dark, so a high value means a dark blob on a brighter background. |
| 18 | i_mean | Average image brightness inside the blob. The image is first normalized by the mean and spread of the solar disk's pixels. |
| 19 | r_disk | Distance of the blob's center from the image center (1024, 1024), divided by 965 pixels, about the Sun's radius. Near 1 means close to the limb. |
| 20 | major | Length of the blob along its main axis (from the spread of its pixel coordinates). |
| 21 | minor | Length across the main axis. |
| 22 | width | Area divided by length: the blob's typical width. |
| 23 | elong | major / minor: how stretched the blob is. |
| 24 | img_n50 | How many blobs the whole image has at t = 0.50. |
| 25 | img_pmean | Average map probability over the whole image. |
| 26 | sib_n | How many children the blob's parent has, including this blob (0 if it has no parent). |
| 27 | rank_in_img | Where the blob's p_mean ranks among all blobs in the image, scaled 0-1 (0 = most confident). |

## Columns 28-44: agreement with I6

The code matches the candidate against every I6 instance by IoU, then summarizes. An I6 instance's score is its box
score times its average mask probability.

| # | Name | What it measures |
|---|---|---|
| 28 | i6_x_best_iou | The highest IoU with any I6 instance. |
| 29 | i6_x_best_score | The score of that best-overlapping instance. |
| 30 | i6_x_best_det | Its box score alone. |
| 31 | i6_x_score_ov50 | The highest score among instances that overlap the candidate at IoU > 0.5, the competition's own match rule (0 if none). |
| 32 | i6_x_n_ov30 | How many instances overlap the candidate at IoU > 0.3. |
| 33 | i6_x_cov_sel | Best IoU with I6's own "answer": the set of non-overlapping instances you get by taking them in order of score (stopping below 0.5), each keeping its unclaimed part if that's at least 70% of the mask and 20 pixels. |
| 34 | i6_r_ov_score_max | The highest IoU × score over all instances. |
| 35 | i6_r_ov_score_sum | The sum of IoU × score over all instances. |
| 36 | i6_r_score_ov3050 | The best score among near misses (IoU 0.3-0.5). |
| 37 | i6_r_score_ov1030 | The best score among weak overlaps (IoU 0.1-0.3). |
| 38 | i6_r_best_mprob | The best-matching instance's average mask probability. |
| 39 | i6_r_cov_mine | The share of the candidate that lies inside its best-matching instance. |
| 40 | i6_r_cov_ext | The share of that instance that lies inside the candidate. |
| 41 | i6_r_inside_score | The total score of instances that lie more than 80% inside the candidate. A high value hints that the candidate merges several filaments. |
| 42 | i6_r_inside_n | How many instances lie more than 80% inside the candidate. |
| 43 | i6_r_best_rank | The best-matching instance's score rank within the image, scaled 0-1 (0 = top). |
| 44 | i6_r_is_best_match | 1 if no plain cnx blob matches that instance better than this candidate does, else 0. |

## Columns 45-61: agreement with Mask R-CNN

The same 17 measures as columns 28-44, computed against Mask R-CNN's instances (prefix `det_` instead of `i6_`).
Mask R-CNN has a single score per instance, so `det_x_best_det` (47) always equals `det_x_best_score` (46). On the
training images, each candidate gets these columns from the one Mask R-CNN fold copy that never saw its image. On test
images all four fold copies are clean, so each candidate is scored four times, once per copy, and the four scores are
averaged.

## Columns 62-63: improvement over the plain blobs

| # | Name | What it measures |
|---|---|---|
| 62 | d_i6 | The candidate's best IoU with an I6 instance, minus the best IoU any plain cnx blob reaches with that same instance. Above 0 means this candidate fits the instance better than any plain blob. |
| 63 | d_det | The same, for Mask R-CNN. |

## Columns 64-73: what kind of candidate it is

| # | Name | What it measures |
|---|---|---|
| 64 | is_mg | 1 if the candidate is a closing merge: nearby blobs joined by a morphological closing with a 6 or 12 pixel radius. |
| 65 | is_pm | 1 if it's a pairwise merge: two blobs at most 32 pixels apart, joined by a straight bridge. |
| 66 | is_cnxL | 1 if it's a blob of the cnxL map. |
| 67 | is_R34 | 1 if it's a blob of the R34 map. |
| 68-73 | fam_extra0-5 | Six numbers that describe a merge, 0 for every other candidate. For a closing merge: log(merged area / members' area), members' area / merged area, the number of blobs joined and the closing radius, then two zeros. For a pairwise merge: the gap in pixels, the average map probability on the bridge, how well the two blobs' main axes line up, how well the gap lines up with them, the smaller blob's area divided by the larger one's, and log((bridge pixels + 1) / area). |

All four indicators are 0 for a plain cnx blob.

## Columns 74-85: second opinions from the other maps

Each candidate is also checked against the two U-Net maps it did not come from. Slot A and slot B are those two maps:

| Candidate comes from | Slot A | Slot B |
|---|---|---|
| the cnx map (plain blobs and both merge types) | R34 | cnxL |
| the cnxL map | R34 | cnx |
| the R34 map | cnxL | cnx |

Six measures per slot (columns 74-79 for slot A, 80-85 for slot B):

| Name | What it measures |
|---|---|
| xm_mean | Average probability of the other map inside the candidate. |
| xm_max | Highest probability of the other map inside the candidate. |
| xm_frac50 | The share of the candidate's pixels where the other map is above 0.5. |
| xm_edge | Average probability of the other map on the candidate's 1-pixel boundary. |
| xm_iou_same | IoU with the other map's best-overlapping blob, cut at the same threshold level. |
| xm_best_iou | The best such IoU over all 16 levels. |

A candidate's own map is left out on purpose. Against its own map, xm_iou_same and xm_best_iou are always 1, so those
columns would only flag where the candidate came from.
