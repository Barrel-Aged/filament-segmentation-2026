"""G4 supplied-instance sampling and deterministic ROI assembly (CPU only)."""
import math
import cv2
import numpy as np
from pycocotools import mask as coco
from .f2_data import F2Patches, whole_crop
from .dataset import aligned_geometry
from .metrics import encode_mask


def box_iou(a, b):
    a=np.asarray(a,dtype=np.float32).reshape(-1,4);b=np.asarray(b,dtype=np.float32).reshape(-1,4)
    intersection=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    return intersection/np.maximum(1e-8,(a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-intersection)


def nms(boxes,scores,threshold=.7,limit=128):
    """Stable CPU NMS; original order breaks exact score ties."""
    order=sorted(range(len(boxes)),key=lambda i:(-float(scores[i]),i));keep=[]
    while order and len(keep)<limit:
        i=order.pop(0);keep.append(i)
        if order:
            overlap=box_iou([boxes[i]],np.asarray(boxes)[order])[0]
            order=[j for j,v in zip(order,overlap) if v<=threshold]
    return keep


def roi_bounds(box,shape):
    x1,y1,x2,y2=map(float,box)
    side=min(min(shape),max(128,int(math.ceil(max(x2-x1,y2-y1)*1.5))))
    left=int(np.clip(round((x1+x2-side)/2),0,shape[1]-side))
    top=int(np.clip(round((y1+y2-side)/2),0,shape[0]-side))
    return top,left,side


def roi_input(raw,box):
    top,left,side=roi_bounds(box,raw.shape);hint=np.zeros((side,side),np.float32)
    x1,y1,x2,y2=map(float,box)
    xa,ya=max(0,int(np.floor(x1-left))),max(0,int(np.floor(y1-top)))
    xb,yb=min(side,int(np.ceil(x2-left))),min(side,int(np.ceil(y2-top)))
    hint[ya:yb,xa:xb]=1
    image=cv2.resize(raw[top:top+side,left:left+side].astype(np.float32)/255,(512,512),interpolation=cv2.INTER_LINEAR)
    hint=cv2.resize(hint,(512,512),interpolation=cv2.INTER_NEAREST)
    return np.stack([image,hint]),(top,left,side)


def selected_component(binary,hint):
    """Select most box-supported component, then largest; scan order breaks ties."""
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary.astype(np.uint8),8)
    if count==1:return np.zeros(binary.shape,bool)
    support=np.bincount(labels[hint.astype(bool)].ravel(),minlength=count)
    chosen=max(range(1,count),key=lambda i:(int(support[i]),int(stats[i,cv2.CC_STAT_AREA]),-i))
    return labels==chosen


def mask_candidates(boxes,scores,probabilities,shape,threshold):
    result=[]
    for i,(box,score,probability) in enumerate(zip(boxes,scores,probabilities)):
        top,left,side=roi_bounds(box,shape)
        p=cv2.resize(probability.astype(np.float32),(side,side),interpolation=cv2.INTER_LINEAR)
        hint=np.zeros((side,side),np.uint8)
        x1,y1,x2,y2=box
        hint[max(0,int(np.floor(y1-top))):min(side,int(np.ceil(y2-top))),max(0,int(np.floor(x1-left))):min(side,int(np.ceil(x2-left)))]=1
        mask=selected_component(p>=threshold,hint)
        if mask.any():
            full=np.zeros(shape,bool);full[top:top+side,left:left+side]=mask
            result.append({'index':i,'score':float(score)*float(p[mask].mean()),'mask':full,'area':int(mask.sum())})
    return result


def assemble(candidates,score_threshold,area,ownership=True):
    selected=[];occupied=None;rejected={'confidence':0,'area':0,'duplicate':0,'ownership':0}
    for row in sorted(candidates,key=lambda r:(-r['score'],r['index'])):
        if row['score']<score_threshold:rejected['confidence']+=1;continue
        mask=row['mask']
        if mask.sum()<area:rejected['area']+=1;continue
        # Compare original candidate with already accepted, disjoint masks.
        if any(np.count_nonzero(mask&other)/np.count_nonzero(mask|other)>.5 for other in selected):rejected['duplicate']+=1;continue
        if occupied is None:occupied=np.zeros(mask.shape,bool)
        if ownership:
            available=mask&~occupied
            available=selected_component(available,available)
            if available.sum()<area:rejected['ownership']+=1;continue
        else:available=mask
        selected.append(available);occupied|=available
    return [encode_mask(m) for m in selected],rejected


class G4Samples(F2Patches):
    def __init__(self,index,stage,seed=20261014,limit_names=None):
        super().__init__(index,'balanced',seed,limit_names);self.stage=stage

    def sample(self,item):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,404,item]))
        name=self.names[int(rng.integers(len(self.names)))];rids=sorted(self.index.images[name]['annotator_image_ids']);rid=rids[int(rng.integers(len(rids)))];chosen=None
        if rng.random()<(.5 if self.stage=='detector' else .75):
            bins=[k for k in self.categories if self.categories[k]];category=bins[int(rng.integers(len(bins)))];names=sorted(self.categories[category]);name=names[int(rng.integers(len(names)))];rids=sorted(self.categories[category][name]);rid=rids[int(rng.integers(len(rids)))];objects=self.categories[category][name][rid];chosen=objects[int(rng.integers(len(objects)))]
        raw=self.images[name];rotation=int(rng.integers(4));flip=bool(rng.integers(2));fraction=None
        if self.stage=='detector':
            if chosen:top,left,side=whole_crop(chosen['bbox'],raw.shape,rng,1024)
            else:
                disk=self.disks[name];angle=rng.uniform(0,2*np.pi);radius=np.sqrt(rng.random())*disk['radius']*.98
                top=int(np.clip(round(disk['center_y']+radius*np.sin(angle)-512),0,1024));left=int(np.clip(round(disk['center_x']+radius*np.cos(angle)-512),0,1024));side=1024
            image=cv2.resize(raw[top:top+side,left:left+side].astype(np.float32)/255,(1024,1024),interpolation=cv2.INTER_AREA)
            image=aligned_geometry(np.repeat(image[None],3,axis=0),rotation,flip);boxes=[];ids=[]
            for obj in self.instances[rid]:
                yy,xx=np.divmod(obj['pixels'],raw.shape[1]);valid=(yy>=top)&(yy<top+side)&(xx>=left)&(xx<left+side)
                if chosen and obj['id']==chosen['id']:fraction=float(valid.mean())
                if valid.sum()<4:continue
                # Raster transformation matches image D4 exactly, including clipping.
                mask=np.zeros((side,side),np.uint8);mask[yy[valid]-top,xx[valid]-left]=1
                if side!=1024:mask=cv2.resize(mask,(1024,1024),interpolation=cv2.INTER_NEAREST)
                mask=aligned_geometry(mask[None],rotation,flip)[0];ys,xs=np.nonzero(mask)
                if len(xs):boxes.append([int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]);ids.append(obj['id'])
            target={'boxes':np.asarray(boxes,np.float32).reshape(-1,4),'labels':np.ones(len(boxes),np.int64)}
            extra={'instance_ids':ids}
        else:
            if chosen:
                x,y,w,h=chosen['bbox'];cx=x+w/2+rng.uniform(-.15,.15)*w;cy=y+h/2+rng.uniform(-.15,.15)*h;w=max(2,w*rng.uniform(.8,1.2));h=max(2,h*rng.uniform(.8,1.2));box=np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2]);box[[0,2]]=np.clip(box[[0,2]],0,raw.shape[1]);box[[1,3]]=np.clip(box[[1,3]],0,raw.shape[0])
            else:
                # Annotation-relative empty context; never declared a true defect.
                for attempt in range(100):
                    side=int(rng.choice([128,256,384,512]));top=int(rng.integers(0,raw.shape[0]-side+1));left=int(rng.integers(0,raw.shape[1]-side+1));box=np.array([left+side/6,top+side/6,left+side*5/6,top+side*5/6])
                    rt,rl,rs=roi_bounds(box,raw.shape)
                    if not self.targets[rid][rt:rt+rs,rl:rl+rs].any():break
                else:raise RuntimeError('No annotation-relative empty ROI after 100 attempts')
            image,(top,left,side)=roi_input(raw,box);mask=np.zeros((side,side),np.uint8)
            if chosen:
                yy,xx=np.divmod(chosen['pixels'],raw.shape[1]);valid=(yy>=top)&(yy<top+side)&(xx>=left)&(xx<left+side);mask[yy[valid]-top,xx[valid]-left]=1;fraction=float(valid.mean())
            target=aligned_geometry(cv2.resize(mask,(512,512),interpolation=cv2.INTER_NEAREST)[None].astype(np.float32),rotation,flip)
            image=aligned_geometry(image,rotation,flip);extra={'hint_box':box.tolist()}
        metadata={'item':item,'stage':self.stage,'file_name':name,'annotator_image_id':rid,'fold':self.index.folds[name],'source_sha256':self.index.images[name]['sha256'],'target_sha256':self.target_hashes[rid],'selected_instance_id':chosen['id'] if chosen else None,'size_bin':chosen['bin'] if chosen else None,'top':top,'left':left,'native_side':side,'rotations_90':rotation,'horizontal_flip':flip,'selected_fraction':fraction,**extra}
        return np.ascontiguousarray(image),target,metadata
