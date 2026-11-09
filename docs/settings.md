# Training and inference settings

The four-page report doesn't have room for every setting, so this page lists them. The exact arguments of every
trained network are in `configs/`, and the code is in `training/` and `filseg/`.

## Image preprocessing

Each image is read as a 2048 × 2048 uint8 array with OpenCV. The solar disk is the set of pixels above 30% of the
image's 99th percentile. The U-Nets see `clip((x − μ) / σ, −6, 4)`, where μ and σ are the disk's mean and standard
deviation. All networks see full-resolution images. Mask R-CNN needs three channels, so it gets the image divided by
255, repeated three times.

## U-Nets (12 models: cnx, cnxL and R34, 4 fold copies each)

| Setting | Value |
|---|---|
| Input and output | one input channel; two sigmoid outputs, a filament mask and a 3-pixel spine |
| cnx | ConvNeXt-T encoder, decoder channels (256, 128, 64, 32, 16), 24,000 steps |
| cnxL | the same network, 48,000 steps |
| R34 | ResNet34 encoder, the library's default decoder, 30,000 steps |
| Starting weights | ImageNet, for all encoders |
| Mask target | the mean of the image's annotator masks (a soft target) |
| Spine target | the mean of the annotators' 3-pixel spine lines |
| Crops | 512 pixels; 60% centered on a random instance with jitter SD 77 pixels, 40% anywhere on the disk |
| Augmentation | rotations, scale 0.9-1.1, flips, blur and JPEG recompression (30% each), contrast × 0.85-1.15, offset ± 0.15 |
| Loss | BCE + 0.5 × soft Dice on the mask, plus 0.3 × BCE on the spine |
| Optimizer | AdamW, weight decay 10⁻², learning rate 2 × 10⁻⁴ (encoder) and 10⁻³ (decoder) |
| Schedule | batch 8, 500 warm-up steps, cosine decay to 1% |
| Other | bfloat16 autocast, gradient-norm clip 1, seed 0 |
| Saved model | the final weight average (EMA, decay 0.999), with no checkpoint selection |
| Inference | the 8 rotation and flip views and the 4 fold copies are averaged and stored as round(255p) |

## Mask R-CNN (4 models, one per fold)

| Setting | Value |
|---|---|
| Model | torchvision `maskrcnn_resnet50_fpn_v2`, COCO starting weights, new two-class heads |
| Training sample | one random annotator's drawing of a random image |
| Crops | 1024 pixels; 70% centered on an instance, jitter SD 205 pixels |
| Augmentation | the 8 rotation and flip transforms, gamma 0.8-1.25, gain 0.9-1.1, blur 30% |
| Optimizer | SGD, momentum 0.9, weight decay 10⁻⁴, learning rate 0.01 |
| Schedule | 12,000 iterations of batch 4, 500 warm-up steps, cosine decay to 1%, gradient-norm clip 10, seed 0 |
| Inference | full resolution in float32 (under bfloat16, torchvision decodes boxes on an 8-pixel grid at this size), at most 100 detections, score above 0.03, mask probability above 0.5, at least 20 pixels |

## I6 detector and refiner

| Setting | Value |
|---|---|
| Detector | Faster R-CNN R50-FPN v2, COCO starting weights, new two-class box head |
| Tiles | 1024 pixels at offsets 0, 768 and 1024 |
| Anchors | 16-256 pixels, aspect ratios 0.25, 1 and 4 |
| Detections | score floor 0.01, NMS 0.7, at most 128 boxes per image |
| Refiner | ResNet18 U-Net, ImageNet starting weights, two input channels (the image and a box hint), 512-pixel crop around each box with context 1.5 × the box and at least 128 pixels |
| Mask | probability above 0.5; the 8-connected component with the most pixels inside the box is kept |
| Candidate score | detector score × mean mask probability |
| Training data | the production models used all 707 images, seed 20261014; both run in bfloat16 |

## Candidates

The candidates are every 8-connected component of `P > round(255t)` on the cnx map, for the 16 levels t = 0.10, 0.15,
..., 0.85, with at least 40 pixels. The pool also has components of a morphological closing (disk radius 6 or 12
pixels) that join at least two same-level blobs, pairs of same-level blobs within 32 pixels joined by a straight
bridge, and the blobs of the cnxL and R34 maps. New candidates under 40 pixels, exact pixel duplicates and candidates
with IoU of at least 0.9 to an existing one are dropped.

## Scorers

| Setting | LightGBM | ExtraTrees |
|---|---|---|
| Models | seeds 10-19 | seeds 10-19 |
| Training rows | 292,535 rows from 707 images | 148,711 rows from 566 images |
| Size | 600 rounds, learning rate 0.05, 31 leaves | 60 trees, no bootstrap |
| Leaf and sampling | at least 100 rows per leaf, feature and bagging fraction 0.8, L2 penalty 1 | at least 100 rows per leaf, half the features per split |

The target and weights are described in `training/scorer_data/README.md`. The blend is q = 0.7 × LightGBM + 0.3 ×
ExtraTrees.

## Selection, shift and writing

The selection score is s = q − 0.2·|t − 0.475| − 0.03·[not a plain cnx blob], and only candidates with s above
τ = 0.24 can be picked. For each picked blob of the cnx, cnxL or R34 map, the code crops the blob's own map to its
bounding box plus 4 pixels, shifts it by (−0.23, 0) pixels with linear interpolation (SciPy, float32), and keeps the
pixels strictly above round(255t) inside the 3 × 3 dilation of the stored mask. Merged candidates keep their masks.
Masks then claim pixels in descending s. Instances are written as column-major COCO run-length encoding with
pycocotools.
