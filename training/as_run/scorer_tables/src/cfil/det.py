"""Instance model: torchvision Mask R-CNN (R50-FPN-v2, COCO init) trained on random-annotator instances (remote GPU only).

A second, object-level source for the stacking scorer. Outputs per image the same JSON layout as the Codex candidate
roots read by ext.py: {'views': {'0.5': rows}, 'predictions': {'set_selector_pq': selected}} plus manifest.json.

usage: det.py train --name N --fold F [--iters 12000] [--accum 1] [--device cuda]
       det.py infer --name N --split F|test --out DIR [--device cuda] [--limit N]

Local overnight GPU runs (user-approved, CFIL_LOCAL_GPU=1, see cfil.guard; work/ideas13/night/run_night.sh) set CFIL_ROOT
(cache/, runs/), CFIL_ANN and TORCH_HOME; unset, everything is the remote layout as before. Training writes a resumable
runs/<name>/ckpt.pt (model, optimizer, step, RNG) every 500 steps and resumes from it while done.json is missing; optimizer
step s always consumes samples (s-1)*bs*accum ... s*bs*accum-1 of the index-deterministic Crops stream.
Optional env (unset = off): CFIL_STOP_AT=<unix s>: training pauses at the first step boundary past it (ckpt.pt, exit 3)
and inference exits 3 before its next image (no manifest); CFIL_STOP_FILE=<path>: the same as soon as that file exists
(manual pause); CFIL_SLOW_SITS=<s/it>: training exits 4 if its mean s/it over the first 200 steps of the session is above
it (Windows can spill VRAM into system RAM instead of raising out-of-memory); HF_HUB_OFFLINE=1: refuse to download the
COCO weights; CFIL_DET_TRACE=1: per-micro-batch trace on stdout.
"""
import argparse, contextlib, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfil import guard  # noqa: F401  (host guard before torch)
import numpy as np, cv2, torch
from pycocotools import mask as mu
from cfil.detrec import build_record

ROOT = os.environ.get('CFIL_ROOT', '/home/barrelaged/projects/filament-claude')  # local GPU runs: claude/work
CACHE = os.path.join(ROOT, 'cache')
ANN = os.environ.get('CFIL_ANN', os.path.join(ROOT, 'data', 'annotations', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
os.environ.setdefault('TORCH_HOME', os.path.join(ROOT, 'cache', 'torch'))  # keep weight downloads inside the project
H = W = 2048
CKPT_EVERY = 500
RESUME_KEYS = ('fold', 'iters', 'bs', 'accum', 'crop', 'lr', 'warmup', 'seed', 'init', 'mask_only', 'mask56', 'target')


def build(pretrained=True, mask56=False):
    """mask56: mask RoIAlign on P2 only (stride 4) at 28x28, so the mask head outputs 56x56 masks (same parameters)."""
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops import MultiScaleRoIAlign
    wts = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    if wts is not None and os.environ.get('HF_HUB_OFFLINE') == '1':  # offline runs must find the file, never download it
        f = os.path.join(torch.hub.get_dir(), 'checkpoints', os.path.basename(wts.url))
        if not os.path.isfile(f):
            raise RuntimeError(f'offline (HF_HUB_OFFLINE=1): COCO weights missing at {f}; refusing to download')
    m = maskrcnn_resnet50_fpn_v2(weights=wts, min_size=1024, max_size=1024, box_detections_per_img=100, box_score_thresh=0.03)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    if mask56:
        m.roi_heads.mask_roi_pool = MultiScaleRoIAlign(['0'], 28, sampling_ratio=2)
    return m


def record_rles():
    """{record id: [RLE per instance]} at full resolution."""
    ann = json.load(open(ANN))
    out = {}
    for a in ann['annotations']:
        segs = json.loads(a['segmentation']) if isinstance(a['segmentation'], str) else a['segmentation']
        segs = [s for s in segs if len(s) >= 6]
        if segs:
            out.setdefault(a['image_id'], []).append(mu.merge(mu.frPyObjects(segs, H, W)))
    return out


class Crops(torch.utils.data.Dataset):
    def __init__(self, idx, meta, rles, crop, length, seed, target='random'):
        self.idx, self.meta, self.rles, self.crop, self.length, self.seed = list(idx), meta, rles, crop, length, seed
        self.target = target  # 'random': one random annotator per sample; 'union': every annotator's instances
        self.img = None

    def __len__(self):
        return self.length

    def __getstate__(self):  # spawn workers (Windows) re-open the memmap instead of receiving a pickled copy of it
        return {**self.__dict__, 'img': None}

    def __getitem__(self, k):
        if self.img is None:
            self.img = np.load(os.path.join(CACHE, 'train_img.npy'), mmap_mode='r')
        rng = np.random.default_rng((self.seed, k))
        i = self.idx[rng.integers(len(self.idx))]
        m = self.meta[i]
        rec = m['records'][rng.integers(len(m['records']))]  # random annotator: scores estimate P(an annotator draws it)
        rl = self.rles.get(rec, []) if self.target == 'random' else [r_ for rr in m['records'] for r_ in self.rles.get(rr, [])]
        C = self.crop
        if rl and rng.random() < 0.7:
            x, y, w, h = mu.toBbox(rl[rng.integers(len(rl))])
            cx, cy = x + w / 2 + rng.normal(0, C * 0.2), y + h / 2 + rng.normal(0, C * 0.2)
        else:
            r = 965 * math.sqrt(rng.random()); t = rng.random() * 2 * math.pi
            cx, cy = 1024 + r * math.cos(t), 1024 + r * math.sin(t)
        x0 = int(np.clip(round(cx - C / 2), 0, W - C)); y0 = int(np.clip(round(cy - C / 2), 0, H - C))
        img = np.asarray(self.img[i, y0:y0 + C, x0:x0 + C]).astype(np.float32) / 255
        masks = []
        for r_ in rl:
            bx, by, bw, bh = mu.toBbox(r_)
            if bx > x0 + C or by > y0 + C or bx + bw < x0 or by + bh < y0:
                continue
            mm = mu.decode(r_)[y0:y0 + C, x0:x0 + C]
            if mm.sum() >= 30:
                masks.append(mm)
        k_rot = int(rng.integers(4)); flip = rng.random() < 0.5
        tf = lambda a: (np.rot90(a, k_rot)[:, ::-1] if flip else np.rot90(a, k_rot))
        img = tf(img)
        img = np.clip(img ** rng.uniform(0.8, 1.25) * rng.uniform(0.9, 1.1), 0, 1)
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(np.ascontiguousarray(img), (0, 0), rng.uniform(0.3, 1.2))
        masks = [np.ascontiguousarray(tf(mm)) for mm in masks]
        boxes = []
        for mm in masks:
            ys, xs = np.nonzero(mm)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        x = torch.from_numpy(np.ascontiguousarray(img))[None].expand(3, -1, -1).contiguous()
        tgt = {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
               'labels': torch.ones(len(masks), dtype=torch.int64),
               'masks': torch.from_numpy(np.stack(masks)) if masks else torch.zeros((0, C, C), dtype=torch.uint8)}
        return x, tgt


def collate(b):  # module level: Windows spawn workers cannot pickle a lambda
    return tuple(zip(*b))


def _save(obj, path):
    """torch.save to a temporary file, then rename over the target: a crash never leaves a truncated file behind."""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    for k in range(10):
        try:
            os.replace(tmp, path); return
        except PermissionError:  # Windows: a scanner can hold the target open for a moment
            if k == 9:
                raise
            time.sleep(1)


def _mem(dev):
    if dev.type != 'cuda':
        return ''
    return f' mem {torch.cuda.max_memory_allocated()/2**30:.2f}G res {torch.cuda.max_memory_reserved()/2**30:.2f}G'


def _stop():
    """Why GPU work must stop now ('' = carry on): the CFIL_STOP_AT deadline passed or the CFIL_STOP_FILE exists."""
    at, f = float(os.environ.get('CFIL_STOP_AT') or 0), os.environ.get('CFIL_STOP_FILE')
    return 'CFIL_STOP_AT deadline' if at and time.time() >= at else 'CFIL_STOP_FILE' if f and os.path.exists(f) else ''


def train(a):
    slow = float(os.environ.get('CFIL_SLOW_SITS') or 0)
    trace = os.environ.get('CFIL_DET_TRACE') == '1'
    if _stop():
        print(f'{_stop()}: not starting', flush=True); sys.exit(3)
    dev = torch.device(a.device)
    meta = json.load(open(os.path.join(CACHE, 'meta.json')))['train']
    idx = [i for i, m in enumerate(meta) if m['fold'] != 4 and m['fold'] != a.fold]
    out = os.path.join(ROOT, 'runs', a.name); os.makedirs(out, exist_ok=True)
    ck_path, ck = os.path.join(out, 'ckpt.pt'), None
    if os.path.exists(ck_path) and not os.path.exists(os.path.join(out, 'done.json')):
        ck = torch.load(ck_path, map_location='cpu', weights_only=True)
        want = {k: vars(a)[k] for k in RESUME_KEYS}
        if ck['args'] != want:
            raise SystemExit(f'{ck_path} was written with {ck["args"]}, not {want}: delete it to start over')
    json.dump(vars(a) | {'n_train_images': len(idx)}, open(os.path.join(out, 'args.json'), 'w'), indent=1)
    torch.manual_seed(a.seed)
    model = build(pretrained=not a.init, mask56=a.mask56)
    if a.init:  # continue from another run's weights (e.g. det-f0 for a mask-branch fine-tune)
        model.load_state_dict(torch.load(os.path.join(ROOT, 'runs', a.init, 'model.pt'), map_location='cpu'))
    if a.mask_only:  # train only the mask branch; boxes and scores stay exactly those of the init run
        for n_, p_ in model.named_parameters():
            p_.requires_grad = n_.startswith(('roi_heads.mask_head.', 'roi_heads.mask_predictor.'))
        model.roi_heads.fg_bg_sampler.batch_size_per_image = 256  # <= 64 positive mask ROIs per image
    model = model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    start, sec0 = 0, 0.0
    if ck is not None:  # model weights incl. BatchNorm buffers, SGD momentum, step, training seconds so far
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); start, sec0 = ck['step'], ck['sec']
        ck = {'rng_cpu': ck['rng_cpu'], 'rng_cuda': ck.get('rng_cuda')}  # release the CPU copies of the weights
    n = a.bs * a.accum  # samples per optimizer step
    ds = Crops(idx, meta, record_rles(), a.crop, a.iters * n, a.seed, a.target)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, collate_fn=collate,
                                     sampler=range(start * n, len(ds)) if start else None,
                                     persistent_workers=True, prefetch_factor=4)
    log = open(os.path.join(out, 'log.txt'), 'a'); t0 = time.time()
    model.train()
    if a.mask_only:  # frozen BatchNorm statistics everywhere outside the mask head
        for n_, mod in model.named_modules():
            if isinstance(mod, torch.nn.BatchNorm2d) and not n_.startswith('roi_heads.mask_head'):
                mod.eval()
    amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if dev.type == 'cuda' else contextlib.nullcontext
    it = iter(dl)
    if ck is not None:  # after the loader's base-seed draw, so the model's random stream continues where it stopped
        torch.set_rng_state(ck['rng_cpu'])
        if dev.type == 'cuda' and ck.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(ck['rng_cuda'])
        msg = f'resume at step {start} (next sample {start * n}, {sec0:.0f}s trained so far)'
        print(msg, flush=True); log.write(msg + '\n'); log.flush()

    def save_ckpt(step):
        _save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step, 'sec': sec0 + time.time() - t0,
               'args': {k: vars(a)[k] for k in RESUME_KEYS}, 'rng_cpu': torch.get_rng_state(),
               'rng_cuda': torch.cuda.get_rng_state_all() if dev.type == 'cuda' else None}, ck_path)

    def leave(code, msg):
        print(msg, flush=True); log.write(msg + '\n'); log.close(); sys.exit(code)

    for step in range(start + 1, a.iters + 1):
        f = step / a.warmup if step < a.warmup else 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / (a.iters - a.warmup)))
        for g in opt.param_groups:
            g['lr'] = a.lr * max(f, 0.01)
        opt.zero_grad(set_to_none=True)
        ok, seen = True, []
        for j in range(a.accum):  # micro-batches of one optimizer step, consecutive in the sample stream
            xs, ts = next(it)
            if trace:
                lo = (step - 1) * n + j * a.bs
                print(f'trace step {step} micro {j} lr {opt.param_groups[0]["lr"]:.6g} samples {lo}-{lo + len(xs) - 1} '
                      f'xsum {[round(float(x.double().sum()), 4) for x in xs]} ninst {[len(t["boxes"]) for t in ts]}', flush=True)
            if not ok:  # a skipped step still consumes all its samples, so the stream stays aligned with the step
                continue
            xs = [x.to(dev, non_blocking=True) for x in xs]
            ts = [{k: v.to(dev, non_blocking=True) for k, v in t.items()} for t in ts]
            with amp():
                losses = model(xs, ts)
            loss = losses['loss_mask'].float() if a.mask_only else sum(v.float() for v in losses.values())
            if not torch.isfinite(loss):
                print('non-finite loss, skipping', step, flush=True); opt.zero_grad(set_to_none=True); ok = False; continue
            (loss / a.accum if a.accum > 1 else loss).backward()
            seen.append((loss.detach(), {k: v.detach().float() for k, v in losses.items()}))
            if trace:
                print(f'trace step {step} micro {j} loss {loss.item():.4f} ' +
                      ' '.join(f'{k[5:]}={v.item():.4f}' for k, v in losses.items()), flush=True)
        if ok:
            torch.nn.utils.clip_grad_norm_(params, 10.0); opt.step()
        if ok and step % 100 == 0:
            lv = sum(l for l, _ in seen) / len(seen); parts = {k: sum(d[k] for _, d in seen) / len(seen) for k in seen[0][1]}
            msg = f'step {step} loss {lv.item():.4f} ' + ' '.join(f'{k[5:]}={v.item():.3f}' for k, v in parts.items()) + \
                  f' {(time.time()-t0)/(step-start):.3f}s/it' + _mem(dev)
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
        if step % 3000 == 0 or step == a.iters:
            _save(model.state_dict(), os.path.join(out, 'model.pt'))
        if step == a.iters:
            break
        if step % CKPT_EVERY == 0:
            save_ckpt(step)
        if slow and step - start == 200 and (time.time() - t0) / 200 > slow:
            leave(4, f'slow: {(time.time() - t0) / 200:.3f}s/it over the first 200 steps of this session (step {step}) '
                     f'> CFIL_SLOW_SITS={slow}' + _mem(dev))
        why = _stop()
        if why:
            save_ckpt(step)
            leave(3, f'paused at step {step} ({why}); next sample {step * n}' + _mem(dev))
    json.dump({'done': True, 'sec': sec0 + time.time() - t0}, open(os.path.join(out, 'done.json'), 'w'))
    for f_ in (ck_path, ck_path + '.tmp'):  # finished: the resumable state is no longer needed
        if os.path.exists(f_):
            os.remove(f_)


@torch.no_grad()
def infer(a):
    dev = torch.device(a.device)
    meta = json.load(open(os.path.join(CACHE, 'meta.json')))
    args = json.load(open(os.path.join(ROOT, 'runs', a.name, 'args.json')))
    model = build(pretrained=False, mask56=args.get('mask56', False))
    model.load_state_dict(torch.load(os.path.join(ROOT, 'runs', a.name, 'model.pt'), map_location='cpu'))
    model.transform.min_size, model.transform.max_size = (H,), H  # full native resolution
    model = model.to(dev).eval()
    if a.split == 'test':
        recs, src = meta['test'], np.load(os.path.join(CACHE, 'test_img.npy'), mmap_mode='r'); ids = list(range(len(recs)))
    else:
        recs, src = meta['train'], np.load(os.path.join(CACHE, 'train_img.npy'), mmap_mode='r')
        ids = [i for i, m in enumerate(recs) if m['fold'] == int(a.split)]
    if a.limit:
        ids = ids[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    if a.dump:  # raw 28x28 mask probabilities + boxes (before pasting), for CPU re-binarisation (det_rebin.py)
        os.makedirs(os.path.join(a.out, 'dump'), exist_ok=True)
        raw = {}
        # copy at hook time: transform.postprocess later replaces the dict's masks with pasted full-size ones in place
        model.roi_heads.register_forward_hook(lambda mod, inp, out: raw.update(r={k: v.detach().clone() for k, v in out[0][0].items()}))
    man = []; t0 = time.time()
    for n, i in enumerate(ids):
        if _stop():
            print(f'{_stop()}: stopped before image {n} of {len(ids)}, no manifest written', flush=True); sys.exit(3)
        x = torch.from_numpy(np.asarray(src[i]).astype(np.float32) / 255)[None].expand(3, -1, -1).to(dev)
        o = model([x])[0]  # fp32: under bf16 autocast torchvision decodes boxes on an 8 px grid at 2048 and clips masks
        sc = o['scores'].float().cpu().numpy(); pm = o['masks'][:, 0].float()
        binm = (pm > 0.5).cpu().numpy(); mprob = [float(pm[j][pm[j] > 0.5].mean()) if binm[j].any() else 0.5 for j in range(len(sc))]
        rows, sel = build_record(zip(binm, mprob), sc)
        if a.dump:
            r = raw['r']
            np.savez_compressed(os.path.join(a.out, 'dump', f'{n:04d}.npz'), boxes=r['boxes'].float().cpu().numpy(),
                                scores=r['scores'].float().cpu().numpy(), masks=r['masks'][:, 0].half().cpu().numpy())
        fn = f'{n:04d}.json'
        json.dump({'views': {'0.5': rows}, 'predictions': {'set_selector_pq': sel}}, open(os.path.join(a.out, fn), 'w'))
        man.append({'file': fn, 'file_name': recs[i]['file_name'], 'kind': 'clean'})
    json.dump({'records': man, 'model': a.name, 'split': a.split, 'sec': time.time() - t0}, open(os.path.join(a.out, 'manifest.json'), 'w'))
    print('done', len(ids), f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--name', required=True); t.add_argument('--fold', type=int, required=True)
    t.add_argument('--iters', type=int, default=12000); t.add_argument('--bs', type=int, default=4)
    t.add_argument('--crop', type=int, default=1024); t.add_argument('--lr', type=float, default=0.01)
    t.add_argument('--warmup', type=int, default=500); t.add_argument('--workers', type=int, default=5)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--init', default=''); t.add_argument('--mask-only', action='store_true')
    t.add_argument('--mask56', action='store_true'); t.add_argument('--target', default='random', choices=['random', 'union'])
    t.add_argument('--accum', type=int, default=1)  # micro-batches of --bs per optimizer step (memory fallback); iters = steps
    t.add_argument('--device', default='cuda')
    i = sub.add_parser('infer')
    i.add_argument('--name', required=True); i.add_argument('--split', required=True); i.add_argument('--out', required=True)
    i.add_argument('--no-dump', dest='dump', action='store_false')  # raw masks are dumped by default (small)
    i.add_argument('--device', default='cuda'); i.add_argument('--limit', type=int, default=0)  # first N images (0 = all)
    a = p.parse_args()
    train(a) if a.cmd == 'train' else infer(a)
