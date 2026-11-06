"""Dense probability maps (stage 2): three U-Nets (cnx, cnxL, R34), four fold models each, averaged.

Every model is an segmentation_models_pytorch U-Net with one input channel (the normalised image) and two output
channels (0 = filament mask, 1 = filament spine):

- ``cnx``  : encoder ``tu-convnext_tiny`` (timm ConvNeXt-T), decoder channels (256, 128, 64, 32, 16), 24,000 steps;
- ``cnxL`` : the same network trained for 48,000 steps;
- ``r34``  : encoder ``resnet34``, smp's default decoder, 30,000 steps.

The architecture of each fold model comes from ``configs/<source>-f<k>.json`` (its training arguments) and the weights
(the final EMA state dict) from ``weights/<source>-f<k>/ema.pt``.

Inference, per image (``infer``):
1. ``normalize``: ``clip((pixel - disk_mean) / disk_std, -6, 4)`` in float32, with the disk statistics of
   ``filseg.data.disk_stats``;
2. ``predict_full``: the whole 2048 x 2048 image goes through the network in each of the 8 D4 views (4 rotations,
   each with and without a horizontal flip); the sigmoids are mapped back and averaged;
3. the 4 fold models are averaged, and the result is stored as ``round(255 * clamp(p, 0, 1))`` uint8.

Output: ``<out>.npy`` (n, 2, 2048, 2048) uint8 and ``<out>.npy.json`` with ``ids``, ``files`` (the image file names,
the order stage 5 reads), ``models`` and ``sec``; the json is written last and marks a finished file.

Device: ``device='cuda'`` runs the network under ``torch.autocast('cuda', dtype=torch.bfloat16)``, as the original
run did; ``device='cpu'`` runs it in float32 (much slower: on a desktop CPU about 15 s per image and model for R34
and about 50 s for ConvNeXt-T; it differs from the bf16 GPU maps by up to a few levels). The maps that made the
submission were made on GPUs (cnx and cnxL on an RTX 4070 SUPER, R34 on an RTX 3070 Ti); a rerun on another GPU or
library build is not bit-identical (bf16 kernels differ), it differs by a few uint8 levels at some pixels.

Changes from the original run
-----------------------------
The arithmetic and its order are those of ``src/cfil/dense.py`` (``build_model`` L16-21, ``normalize`` L24-26,
``predict_full`` L123-137, ``infer`` L140-166).
- No host guard: ``from cfil import guard`` (L4, which raised unless the host was the remote GPU box or
  ``CFIL_LOCAL_GPU=1``) and the ``sys.path`` insert (L3) are gone.
- ``device`` argument: the original hard-coded ``dev = 'cuda'`` (L142). On CUDA the autocast is unchanged (L132); on
  the CPU no autocast is used, so the network runs in float32 (the original had no CPU path).
- Paths: ``ROOT``/``CACHE`` (L8-9, default: the training host's project folder) are gone. The model
  arguments come from ``configs/<name>.json`` (byte copies of ``runs/<name>/args.json``, L145) and the weights from
  ``filseg.weights.open_file('<name>/ema.pt')`` (L147 read ``runs/<name>/ema.pt``); the images and their disk
  statistics are arguments (L141, L149-150 read ``cache/meta.json`` and ``cache/test_img.npy``).
- Test split only: the ``--split <folds>`` branch over the training cache (L152-155) is gone; it made the OOF maps that
  trained the scorer (see ``training/train_dense.py``).
- The sidecar json also records ``device`` and the torch version (stage 5 reads only ``files``).
- Training (``Crops``, ``dice_loss``, ``train``, L29-120) is in ``training/train_dense.py``.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

from filseg import paths, weights

SOURCES = {'cnx': 'testcnx', 'cnxL': 'testcnxL', 'r34': 'testr34'}   # source -> map file stem used by stage 5
FOLDS = (0, 1, 2, 3)


def build_model(arch, encoder, weights='imagenet'):
    """The smp network of one dense model: 1 input channel, 2 output channels (mask, spine).

    ``weights='imagenet'`` loads the pretrained encoder (training); inference passes ``weights=None`` (no download)
    and then loads the EMA state dict.
    """
    cls = {'unet': smp.Unet, 'unetpp': smp.UnetPlusPlus, 'fpn': smp.FPN}[arch]
    kw = dict(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=2)
    if arch == 'unet' and encoder.startswith('tu-convnext'):
        kw['decoder_channels'] = (256, 128, 64, 32, 16)
    return cls(**kw)


def normalize(img_u8, mean, std):
    """float32 ``clip((img - mean) / std, -6, 4)``."""
    x = (img_u8.astype(np.float32) - mean) / std
    return np.clip(x, -6, 4)


def _amp(device):
    """bf16 autocast on CUDA, as the original; nothing (float32) on the CPU."""
    if torch.device(device).type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict_full(model, x):
    """x: (1, 1, H, W) normalised, on the model's device. Mean sigmoid over the 8 D4 views; returns (2, H, W) float32."""
    acc = 0
    for k in range(4):
        for fl in (False, True):
            v = torch.rot90(x, k, (2, 3))
            if fl:
                v = v.flip(3)
            with _amp(x.device):
                o = torch.sigmoid(model(v.contiguous(memory_format=torch.channels_last)).float())
            if fl:
                o = o.flip(3)
            acc = acc + torch.rot90(o, -k, (2, 3))
    return (acc / 8)[0]


def model_args(name: str) -> dict:
    """The recorded training arguments of fold model ``name`` (``configs/<name>.json``)."""
    with open(paths.CONFIGS / f'{name}.json', encoding='utf-8') as fh:
        return json.load(fh)


def load_model(name: str, device='cuda'):
    """Fold model ``name`` (for example ``'cnx-f0'``) in eval mode, channels_last, on ``device``."""
    args = model_args(name)
    m = build_model(args['arch'], args['encoder'], weights=None)
    m.load_state_dict(torch.load(weights.open_file(f'{name}/ema.pt'), map_location='cpu'))
    return m.to(device).eval().to(memory_format=torch.channels_last)


def fold_models(source: str) -> list[str]:
    """The four fold model names of a map source: ``fold_models('cnx') == ['cnx-f0', ..., 'cnx-f3']``."""
    return [f'{source}-f{k}' for k in FOLDS]


@torch.no_grad()
def infer(models, images, recs, out, device='cuda', log=print):
    """Average the D4 maps of ``models`` over the images and write ``out`` (.npy) and ``out + '.json'``.

    models: fold model names (``fold_models(source)``); images: uint8 (n, 2048, 2048) array or memmap;
    recs: per image dicts with ``file_name``, ``disk_mean``, ``disk_std`` (``filseg.data.load_test``);
    out: the .npy path (overwritten). Returns the sidecar dict.
    """
    out = str(out)
    dev = torch.device(device)
    nets = [load_model(name, dev) for name in models]
    ids = list(range(len(recs)))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    res = np.lib.format.open_memmap(out, 'w+', np.uint8, (len(ids), 2, 2048, 2048))
    t0 = time.time()
    for j, i in enumerate(ids):
        x = torch.from_numpy(normalize(np.asarray(images[i]), recs[i]['disk_mean'], recs[i]['disk_std']))[None, None].to(dev)
        p = sum(predict_full(m, x) for m in nets) / len(nets)
        res[j] = (p.clamp(0, 1) * 255).round().byte().cpu().numpy()
        if log and (j % 20 == 0 or j == len(ids) - 1):
            log(f'dense {",".join(models)}: image {j + 1}/{len(ids)} {time.time() - t0:.0f}s')
    res.flush()
    del res
    side = {'ids': ids, 'files': [recs[i]['file_name'] for i in ids], 'models': ','.join(models),
            'sec': time.time() - t0, 'device': str(dev), 'torch': torch.__version__}
    with open(out + '.json', 'w', encoding='utf-8') as fh:
        json.dump(side, fh, indent=1)
    return side


def infer_source(source, images, recs, out_dir=None, device='cuda', log=print):
    """Write ``<out_dir>/test<source>.npy`` (default out_dir ``<work>/maps``) from the four fold models of ``source``."""
    out_dir = Path(out_dir or paths.WORK / 'maps')
    return infer(fold_models(source), images, recs, out_dir / f'{SOURCES[source]}.npy', device=device, log=log)
