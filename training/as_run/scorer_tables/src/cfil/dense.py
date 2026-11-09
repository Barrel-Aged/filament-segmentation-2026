"""Dense soft-target segmentation: training and full-image D4 inference (remote GPU only)."""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfil import guard  # noqa: F401  (host guard before torch)
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp

ROOT = os.environ.get('CFIL_ROOT', '/home/barrelaged/projects/filament-claude')  # local GPU runs: claude/work
CACHE = os.path.join(ROOT, 'cache')


def load_meta():
    return json.load(open(os.path.join(CACHE, 'meta.json')))


def build_model(arch, encoder, weights='imagenet'):
    cls = {'unet': smp.Unet, 'unetpp': smp.UnetPlusPlus, 'fpn': smp.FPN}[arch]
    kw = dict(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=2)
    if arch == 'unet' and encoder.startswith('tu-convnext'):
        kw['decoder_channels'] = (256, 128, 64, 32, 16)
    return cls(**kw)


def normalize(img_u8, mean, std):
    x = (img_u8.astype(np.float32) - mean) / std
    return np.clip(x, -6, 4)


class Crops(torch.utils.data.Dataset):
    def __init__(self, idx, meta, crop, length, seed):
        self.idx, self.meta, self.crop, self.length, self.seed = list(idx), meta, crop, length, seed
        ann = json.load(open(os.path.join(CACHE, 'centers.json')))
        self.centers = [(i, c) for i in self.idx for c in ann[str(i)]]
        self.mm = None

    def __len__(self):
        return self.length

    def _open(self):
        self.mm = {k: np.load(os.path.join(CACHE, f'train_{k}.npy'), mmap_mode='r') for k in ('img', 'soft', 'spine')}

    def __getitem__(self, k):
        if self.mm is None:
            self._open()
        rng = np.random.default_rng((self.seed, k))
        if rng.random() < 0.6:
            i, (cx, cy) = self.centers[rng.integers(len(self.centers))]
            cx += rng.normal(0, self.crop * 0.15); cy += rng.normal(0, self.crop * 0.15)
        else:
            i = self.idx[rng.integers(len(self.idx))]
            r = 965 * math.sqrt(rng.random()); t = rng.random() * 2 * math.pi
            cx, cy = 1024 + r * math.cos(t), 1024 + r * math.sin(t)
        m = self.meta[i]
        ang = rng.uniform(0, 360) if rng.random() < 0.5 else 90 * rng.integers(4)
        sc = rng.uniform(0.9, 1.1)
        C = self.crop
        half = int(C * 0.75 / sc) + 4
        x0, y0 = int(round(cx)) - half, int(round(cy)) - half
        x0 = min(max(x0, 0), 2048 - 2 * half); y0 = min(max(y0, 0), 2048 - 2 * half)
        sl = (i, slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        img, soft, sp = (np.asarray(self.mm[k][sl]) for k in ('img', 'soft', 'spine'))
        M = cv2.getRotationMatrix2D((half, half), ang, sc)
        M[:, 2] += C / 2 - half
        warp = lambda a: cv2.warpAffine(a, M, (C, C), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img, soft, sp = warp(img), warp(soft), warp(sp)
        if rng.random() < 0.5:
            img, soft, sp = img[:, ::-1], soft[:, ::-1], sp[:, ::-1]
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
        if rng.random() < 0.3:
            ok, enc = cv2.imencode('.jpg', np.ascontiguousarray(img), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        x = normalize(img, m['disk_mean'], m['disk_std'])
        x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.15, 0.15)
        y = np.stack([soft, sp]).astype(np.float32) / 255
        return torch.from_numpy(np.ascontiguousarray(x[None])), torch.from_numpy(np.ascontiguousarray(y))


def dice_loss(logit, target):
    p = torch.sigmoid(logit)
    inter = (p * target).sum((1, 2)); den = (p + target).sum((1, 2))
    return (1 - (2 * inter + 1) / (den + 1)).mean()


def train(a):
    meta = load_meta()['train']
    idx = [i for i, m in enumerate(meta) if m['fold'] != 4 and m['fold'] != a.fold]
    out = os.path.join(ROOT, 'runs', a.name); os.makedirs(out, exist_ok=True)
    json.dump(vars(a) | {'n_train_images': len(idx)}, open(os.path.join(out, 'args.json'), 'w'), indent=1)
    torch.manual_seed(a.seed)
    dev = 'cuda'
    model = build_model(a.arch, a.encoder).to(dev).to(memory_format=torch.channels_last)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    enc = [p for n, p in model.named_parameters() if n.startswith('encoder.')]
    dec = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    opt = torch.optim.AdamW([{'params': enc, 'lr': a.lr_enc}, {'params': dec, 'lr': a.lr_dec}], weight_decay=1e-2)
    base = [a.lr_enc, a.lr_dec]
    ds = Crops(idx, meta, a.crop, a.steps * a.bs, a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, persistent_workers=True, prefetch_factor=4)
    t0 = time.time(); log = open(os.path.join(out, 'log.txt'), 'a')
    model.train()
    for step, (x, y) in enumerate(dl, 1):
        f = step / a.warmup if step < a.warmup else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / (a.steps - a.warmup)))
        for g, b in zip(opt.param_groups, base):
            g['lr'] = b * f
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(x).float()
        loss = F.binary_cross_entropy_with_logits(o[:, 0], y[:, 0]) + 0.5 * dice_loss(o[:, 0], y[:, 0]) \
            + 0.3 * F.binary_cross_entropy_with_logits(o[:, 1], y[:, 1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); ema.update_parameters(model)
        if step % 200 == 0:
            msg = f'step {step} loss {loss.item():.4f} lr {f:.3f} {(time.time()-t0)/step:.3f}s/it mem {torch.cuda.max_memory_allocated()/2**30:.2f}G'
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
        if step % 5000 == 0 or step == a.steps:
            torch.save(ema.module.state_dict(), os.path.join(out, 'ema.pt'))
    json.dump({'done': True, 'sec': time.time() - t0}, open(os.path.join(out, 'done.json'), 'w'))


@torch.no_grad()
def predict_full(model, x):
    """x: (1,1,H,W) normalized; mean sigmoid over the 8 D4 views. Returns (2,H,W) float32."""
    acc = 0
    for k in range(4):
        for fl in (False, True):
            v = torch.rot90(x, k, (2, 3))
            if fl:
                v = v.flip(3)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = torch.sigmoid(model(v.contiguous(memory_format=torch.channels_last)).float())
            if fl:
                o = o.flip(3)
            acc = acc + torch.rot90(o, -k, (2, 3))
    return (acc / 8)[0]


def infer(a):
    meta = load_meta()
    dev = 'cuda'
    models = []
    for name in a.models.split(','):
        args = json.load(open(os.path.join(ROOT, 'runs', name, 'args.json')))
        m = build_model(args['arch'], args['encoder'], weights=None)
        m.load_state_dict(torch.load(os.path.join(ROOT, 'runs', name, 'ema.pt'), map_location='cpu'))
        models.append(m.to(dev).eval().to(memory_format=torch.channels_last))
    if a.split == 'test':
        recs, src = meta['test'], np.load(os.path.join(CACHE, 'test_img.npy'), mmap_mode='r')
        ids = list(range(len(recs)))
    else:
        recs, src = meta['train'], np.load(os.path.join(CACHE, 'train_img.npy'), mmap_mode='r')
        folds = [int(f) for f in a.split.split(',')]
        ids = [i for i, m in enumerate(recs) if m['fold'] in folds]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out = np.lib.format.open_memmap(a.out, 'w+', np.uint8, (len(ids), 2, 2048, 2048))
    t0 = time.time()
    for j, i in enumerate(ids):
        x = torch.from_numpy(normalize(np.asarray(src[i]), recs[i]['disk_mean'], recs[i]['disk_std']))[None, None].to(dev)
        p = sum(predict_full(m, x) for m in models) / len(models)
        out[j] = (p.clamp(0, 1) * 255).round().byte().cpu().numpy()
    out.flush()
    json.dump({'ids': ids, 'files': [recs[i]['file_name'] for i in ids], 'models': a.models, 'sec': time.time() - t0},
              open(a.out + '.json', 'w'), indent=1)
    print('done', len(ids), f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--name', required=True); t.add_argument('--fold', type=int, required=True)
    t.add_argument('--arch', default='unet'); t.add_argument('--encoder', default='resnet34')
    t.add_argument('--steps', type=int, default=20000); t.add_argument('--bs', type=int, default=8)
    t.add_argument('--crop', type=int, default=512); t.add_argument('--warmup', type=int, default=500)
    t.add_argument('--lr_enc', type=float, default=2e-4); t.add_argument('--lr_dec', type=float, default=1e-3)
    t.add_argument('--workers', type=int, default=6); t.add_argument('--seed', type=int, default=0)
    i = sub.add_parser('infer')
    i.add_argument('--models', required=True); i.add_argument('--split', required=True); i.add_argument('--out', required=True)
    a = p.parse_args()
    train(a) if a.cmd == 'train' else infer(a)
