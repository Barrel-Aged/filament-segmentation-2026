"""Detector candidates (stage 3): Mask R-CNN (torchvision R50-FPN v2), four fold models, each with its own output.

Each fold model ``det-f<k>`` is a ``maskrcnn_resnet50_fpn_v2`` with new 2-class box and mask predictors, at most 100
detections per image with score above 0.03. Its weights are ``weights/det-f<k>/model.pt`` and its training arguments
``configs/det-f<k>.json`` (only ``mask56`` matters at inference; it is false or absent for all four).

Inference, per image (``infer``): the uint8 image / 255, copied to 3 channels, at native 2048 x 2048 resolution
(torchvision applies the ImageNet mean/std inside the model), always in float32 (under bf16 autocast torchvision
decodes the boxes on an 8 px grid at this size). Each detection's 28 x 28 mask is pasted by torchvision; pixels with
probability > 0.5 form its binary mask. ``build_record`` turns the detections into rows:

- every mask with >= 20 px becomes a row ``{'rle', 'score', 'features': [0, logit(score), logit(mean prob inside)],
  'area', 'rho'}`` (rho = distance of the mask centroid from the disk centre (1024, 1024), / 965 px);
- a disjoint "selected" set: in score order, while score >= 0.5, the part of a mask not yet taken is kept when it
  is >= 70 % of the mask and >= 20 px.

Output folder (the layout stage 5 reads with ``filseg.evidence.load_roots``): ``NNNN.json`` per image =
``{'views': {'0.5': rows}, 'predictions': {'set_selector_pq': selected}}`` and ``manifest.json`` =
``{'records': [{'file', 'file_name', 'kind': 'clean'}], 'model', 'split': 'test', 'sec'}``. With ``dump=True`` also
``dump/NNNN.npz`` with the raw 28 x 28 mask probabilities (fp16), boxes and scores before pasting (not read downstream).

Device: ``'cuda'`` or ``'cpu'``; float32 on both, as in the original run (which used CUDA).

Changes from the original run
-----------------------------
The arithmetic is that of ``src/cfil/det.py`` (``build`` L35-51 without the offline check, ``infer`` L261-300) and
``src/cfil/detrec.py`` (``_rle`` L9-11, ``build_record`` L14-35). The test roots were written on 10-29 by the det.py of
commits 58b3592 / 3516743, whose ``build`` and ``infer`` compute the same thing as the current file (only the
``--device``/``--limit`` options and the stop check were added later, in 1f01db9).
- No host guard: ``from cfil import guard`` (det.py L21) and the ``sys.path`` insert (L20) are gone, and so is the
  import-time ``os.environ.setdefault('TORCH_HOME', ...)`` (L29; only training downloads weights).
- Paths: ``ROOT``/``CACHE``/``ANN`` (L26-28) are gone. Arguments come from ``configs/<name>.json`` (byte copies of
  ``runs/<name>/args.json``, L265), weights from ``filseg.weights.open_file('<name>/model.pt')`` (L267), images and
  file names from arguments (L264, L270-271 read ``cache/meta.json`` and ``cache/test_img.npy``).
- Test split only: the ``--split <fold>`` branch over the training cache (L272-274) is gone; it made the OOF roots
  that trained the scorer.
- ``dump`` defaults to False (the original dumped by default, L278-283, L292-295; nothing downstream reads the dump).
- The ``CFIL_STOP_AT`` / ``CFIL_STOP_FILE`` pause check (L144-147, L285-286) is gone.
- ``detrec.py``'s two functions live here (``_rle``, ``build_record``), unchanged.
- The manifest also records ``device`` and the torch version (stage 5 reads only ``records``).
- ``build``'s ``HF_HUB_OFFLINE`` guard against downloading the COCO weights (L42-45, added after the production runs)
  is not copied; it only matters for training (``training/train_det.py``).
- Training (``record_rles``, ``Crops``, ``train``) is in ``training/train_det.py``.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
from pycocotools import mask as mu

from filseg import paths, weights

H = W = 2048
FOLDS = (0, 1, 2, 3)


def build(pretrained=True, mask56=False):
    """Mask R-CNN R50-FPN v2 with 2-class heads; ``pretrained`` starts from torchvision's COCO weights (training).

    mask56: mask RoIAlign on P2 only (stride 4) at 28x28, so the mask head outputs 56x56 masks (same parameters).
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops import MultiScaleRoIAlign
    m = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
                                 min_size=1024, max_size=1024, box_detections_per_img=100, box_score_thresh=0.03)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    if mask56:
        m.roi_heads.mask_roi_pool = MultiScaleRoIAlign(['0'], 28, sampling_ratio=2)
    return m


def _rle(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    return {'size': [H, W], 'counts': r['counts'].decode()}


def build_record(masks, sc):
    """masks: iterable of (binary mask HxW, mean probability inside it) per detection in score order; sc: scores.
    Returns (rows, selected) in the candidate-root JSON layout read by ``filseg.evidence``."""
    lg = lambda p: math.log(max(p, 1e-6) / max(1 - p, 1e-6))  # noqa: E731
    rows, kept = [], []
    for j, (bm, mp) in enumerate(masks):
        area = int(bm.sum())
        if area < 20:
            continue
        ys, xs = np.nonzero(bm)
        rho = float(np.hypot(xs.mean() - 1024, ys.mean() - 1024) / 965)
        rows.append({'rle': _rle(bm), 'score': float(sc[j]), 'features': [0.0, lg(float(sc[j])), lg(mp)], 'area': area, 'rho': rho})
        kept.append((j, bm))
    # disjoint "selected" set: greedy by score above 0.5, dropping overlaps with pixels already taken
    taken = np.zeros((H, W), bool); sel = []
    for j, bm in kept:
        if sc[j] < 0.5:
            break
        mm = bm & ~taken
        if mm.sum() >= 0.7 * bm.sum() and mm.sum() >= 20:
            taken |= mm; sel.append(_rle(mm))
    return rows, sel


def load_model(name: str, device='cuda'):
    """Fold model ``name`` (for example ``'det-f0'``) in eval mode at native 2048 px resolution, on ``device``."""
    with open(paths.CONFIGS / f'{name}.json', encoding='utf-8') as fh:
        args = json.load(fh)
    model = build(pretrained=False, mask56=args.get('mask56', False))
    model.load_state_dict(torch.load(weights.open_file(f'{name}/model.pt'), map_location='cpu'))
    model.transform.min_size, model.transform.max_size = (H,), H  # full native resolution
    return model.to(device).eval()


@torch.no_grad()
def infer(name, images, recs, out, device='cuda', dump=False, log=print):
    """Write the candidate root of fold model ``name`` for the images into folder ``out``.

    images: uint8 (n, 2048, 2048) array or memmap; recs: per image dicts with ``file_name`` (``filseg.data``);
    out: output folder (files overwritten). Returns the manifest dict.
    """
    out = str(out)
    dev = torch.device(device)
    model = load_model(name, dev)
    ids = list(range(len(recs)))
    os.makedirs(out, exist_ok=True)
    if dump:  # raw 28x28 mask probabilities + boxes (before pasting)
        os.makedirs(os.path.join(out, 'dump'), exist_ok=True)
        raw = {}
        # copy at hook time: transform.postprocess later replaces the dict's masks with pasted full-size ones in place
        model.roi_heads.register_forward_hook(lambda mod, inp, o: raw.update(r={k: v.detach().clone() for k, v in o[0][0].items()}))
    man = []; t0 = time.time()
    for n, i in enumerate(ids):
        x = torch.from_numpy(np.asarray(images[i]).astype(np.float32) / 255)[None].expand(3, -1, -1).to(dev)
        o = model([x])[0]  # fp32: under bf16 autocast torchvision decodes boxes on an 8 px grid at 2048 and clips masks
        sc = o['scores'].float().cpu().numpy(); pm = o['masks'][:, 0].float()
        binm = (pm > 0.5).cpu().numpy(); mprob = [float(pm[j][pm[j] > 0.5].mean()) if binm[j].any() else 0.5 for j in range(len(sc))]
        rows, sel = build_record(zip(binm, mprob), sc)
        if dump:
            r = raw['r']
            np.savez_compressed(os.path.join(out, 'dump', f'{n:04d}.npz'), boxes=r['boxes'].float().cpu().numpy(),
                                scores=r['scores'].float().cpu().numpy(), masks=r['masks'][:, 0].half().cpu().numpy())
        fn = f'{n:04d}.json'
        with open(os.path.join(out, fn), 'w', encoding='utf-8') as fh:
            json.dump({'views': {'0.5': rows}, 'predictions': {'set_selector_pq': sel}}, fh)
        man.append({'file': fn, 'file_name': recs[i]['file_name'], 'kind': 'clean'})
        if log and (n % 20 == 0 or n == len(ids) - 1):
            log(f'det {name}: image {n + 1}/{len(ids)} {time.time() - t0:.0f}s')
    manifest = {'records': man, 'model': name, 'split': 'test', 'sec': time.time() - t0, 'device': str(dev),
                'torch': torch.__version__}
    with open(os.path.join(out, 'manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh)
    return manifest


def infer_all(images, recs, out_root=None, device='cuda', dump=False, log=print):
    """The four fold models' roots ``<out_root>/det-f<k>/test`` (default out_root ``<work>/det``)."""
    out_root = out_root or paths.WORK / 'det'
    return [infer(f'det-f{k}', images, recs, os.path.join(str(out_root), f'det-f{k}', 'test'), device, dump, log)
            for k in FOLDS]
