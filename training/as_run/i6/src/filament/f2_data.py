"""CPU-only nested partitioning and exact, in-memory F2 crop sampling."""
import hashlib
import math
from collections import defaultdict
import cv2
import numpy as np
from pycocotools import mask as coco
from .common import read_json, sha256
from .dataset import ObservationIndex, AnnotationCache, aligned_geometry, crop_origin, estimate_disk
from .metrics import polygon_rle, decode_mask


def nested_partition(split, outer_fold):
    if outer_fold not in split['development_folds']:
        raise ValueError('Development outer fold required')
    rows=[r for r in split['observations'] if r['fold'] in split['development_folds']]
    available=sorted({r['group_id'] for r in rows if r['fold']!=outer_fold},
        key=lambda g:hashlib.sha256(f'F2-inner-v1|{outer_fold}|{g}'.encode()).hexdigest())
    inner=set(available[:math.ceil(len(available)/8)])
    return {kind:sorted(r['file_name'] for r in rows if
        (r['fold']==outer_fold if kind=='outer' else r['fold']!=outer_fold and
         ((r['group_id'] in inner) if kind=='inner' else (r['group_id'] not in inner))))
        for kind in ['fit','inner','outer']}


class F2Index(ObservationIndex):
    def __init__(self, partition, outer_fold):
        super().__init__(validation_fold=outer_fold)
        if partition!=nested_partition(read_json(self.split_path),outer_fold):
            raise ValueError('Nested partition differs from frozen group algorithm')
        self.train_names=partition['fit']
        self.inner_names=partition['inner']
        self.validation_names=partition['outer']


def whole_crop(bbox, shape, rng, patch=512):
    x,y,w,h=map(int,bbox)
    # Exact native crops when possible; larger structures include a 32px margin.
    side=patch if max(w,h)<=patch else min(max(shape),int(math.ceil((max(w,h)+64)/32)*32))
    side=min(side,*shape)
    left=int(rng.integers(max(0,x+w-side),min(x,shape[1]-side)+1))
    top=int(rng.integers(max(0,y+h-side),min(y,shape[0]-side)+1))
    return top,left,side


def lr_factor(step, horizon):
    """LR for the next update after `step` successful updates; never rebound."""
    return .01+.99*.5*(1+math.cos(math.pi*min(step,horizon)/horizon))


class F2Patches:
    """No fitted transforms. Raw/target arrays are exact decoded supplied data.

    Parent process owns the memory bank; num_workers=0 avoids duplicated banks.
    The native sampler preserves the original TrainingPatches RNG sequence.
    """
    def __init__(self,index,sampler,seed,limit_names=None):
        if sampler not in ['native','whole','balanced']:
            raise ValueError('Unknown F2 sampler')
        self.index,self.sampler,self.seed=index,sampler,seed
        self.names=index.train_names if limit_names is None else sorted(limit_names)
        if not set(self.names)<=set(index.train_names):raise ValueError('Training-only bank required')
        self.images={};self.targets={};self.instances={};self.disks={};self.target_hashes={}
        self.categories={k:defaultdict(lambda:defaultdict(list)) for k in ['small','medium','large']}
        cache=AnnotationCache(index)
        for name in self.names:
            image=index.load_image(name);self.images[name]=image;self.disks[name]=estimate_disk(image)
            for rid in sorted(index.images[name]['annotator_image_ids']):
                target,meta=cache.load(rid);self.targets[rid]=target;self.target_hashes[rid]=meta['mask_sha256']
                objects=[]
                for ann in sorted(index.annotations[rid],key=lambda a:a['id']):
                    rle=polygon_rle(ann,*image.shape);pixels=np.flatnonzero(decode_mask(rle))
                    row={'id':ann['id'],'pixels':pixels,'bbox':list(map(int,coco.toBbox(rle))),'area':len(pixels)}
                    row['bin']='small' if len(pixels)<256 else 'medium' if len(pixels)<4096 else 'large'
                    objects.append(row);self.categories[row['bin']][name][rid].append(row)
                self.instances[rid]=objects
        self.bank_bytes=sum(a.nbytes for a in self.images.values())+sum(a.nbytes for a in self.targets.values())+sum(o['pixels'].nbytes for v in self.instances.values() for o in v)

    def __len__(self):return 15000*8

    def __getitem__(self,global_index):
        epoch,item=divmod(global_index,1024)
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,epoch,item]))
        name=self.names[int(rng.integers(len(self.names)))];rids=sorted(self.index.images[name]['annotator_image_ids'])
        rid=rids[int(rng.integers(len(rids)))];objects=self.instances[rid]
        foreground=bool(rng.random()<.5 and objects)
        selected=None;category=None
        if foreground:
            if self.sampler=='balanced':
                bins=[k for k in ['small','medium','large'] if self.categories[k]]
                category=bins[int(rng.integers(len(bins)))];names=sorted(self.categories[category])
                name=names[int(rng.integers(len(names)))];rids=sorted(self.categories[category][name])
                rid=rids[int(rng.integers(len(rids)))];objects=self.categories[category][name][rid]
            selected=objects[int(rng.integers(len(objects)))];category=selected['bin']
            y,x=np.unravel_index(int(selected['pixels'][int(rng.integers(len(selected['pixels'])))]),self.images[name].shape)
        else:
            disk=self.disks[name];angle,radius=rng.uniform(0,2*np.pi),np.sqrt(rng.random())*disk['radius']*.98
            y=int(np.clip(round(disk['center_y']+radius*np.sin(angle)),0,2047))
            x=int(np.clip(round(disk['center_x']+radius*np.cos(angle)),0,2047))
        if foreground and self.sampler!='native':top,left,side=whole_crop(selected['bbox'],self.images[name].shape,rng)
        else:
            top,left=crop_origin(y,x,self.images[name].shape,512,rng);side=512
        raw=self.images[name][top:top+side,left:left+side].astype(np.float32)/255
        target=self.targets[rid][top:top+side,left:left+side].astype(np.float32)
        if side!=512:
            raw=cv2.resize(raw,(512,512),interpolation=cv2.INTER_AREA)
            target=cv2.resize(target,(512,512),interpolation=cv2.INTER_NEAREST)
        rotations=int(rng.integers(4));flip=bool(rng.integers(2))
        fraction=None
        if selected:
            yy,xx=np.divmod(selected['pixels'],2048)
            fraction=float(((yy>=top)&(yy<top+side)&(xx>=left)&(xx<left+side)).mean())
        return {'image':aligned_geometry(raw[None],rotations,flip),
            'target':aligned_geometry(target[None],rotations,flip),
            'metadata':{'file_name':name,'annotator_image_id':rid,'fold':self.index.folds[name],
                'channels':['raw'],'global_index':global_index,'seed':self.seed,'epoch':epoch,'sample_index':item,
                'top':top,'left':left,'patch_size':512,'native_side':side,'scale':512/side,
                'rotations_90':rotations,'horizontal_flip':flip,'sampler':self.sampler,
                'sampling':'foreground' if foreground else 'random_disk','size_bin':category,
                'selected_instance_id':selected['id'] if selected else None,'instance_fraction_in_crop':fraction,
                'source_sha256':self.index.images[name]['sha256'],'target_sha256':self.target_hashes[rid]}}
