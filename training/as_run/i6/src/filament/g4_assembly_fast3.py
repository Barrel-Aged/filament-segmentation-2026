"""Lossless G4 ROI assembly acceleration with exact reference tie fallback."""
import cv2
import numpy as np
from pycocotools import mask as coco
from .g4_data import selected_component
from .metrics import native_rle


def encode_mask(mask):
    assert mask.ndim==2 and mask.dtype==np.bool_
    value=coco.encode(np.asfortranarray(mask,dtype=np.uint8))
    return {'size':list(value['size']),'counts':value['counts'].decode('ascii')}


def assemble(candidates,score_threshold,area,ownership=True):
    selected=[];native_selected=[];occupied=None
    rejected={'confidence':0,'area':0,'duplicate':0,'ownership':0}
    for row in sorted(candidates,key=lambda r:(-r['score'],r['index'])):
        if row['score']<score_threshold:rejected['confidence']+=1;continue
        mask=row['mask']
        cache=row.get('_g4_cached')
        if cache is None or cache['mask'] is not mask:
            mask.flags.writeable=False  # G4 candidate masks are immutable across decoder settings.
            candidate=encode_mask(mask);native=native_rle(candidate)
            cache={'mask':mask,'native':native,'area':int(coco.area(native)),'bbox':list(map(int,coco.toBbox(native)))};row['_g4_cached']=cache
        if cache['area']<area:rejected['area']+=1;continue
        native=cache['native']
        if native_selected and (coco.iou([native],native_selected,[0]*len(native_selected))>.5).any():
            rejected['duplicate']+=1;continue
        if occupied is None:occupied=np.zeros(mask.shape,bool)
        if ownership:
            x,y,w,h=cache['bbox'];crop=mask[y:y+h,x:x+w]&~occupied[y:y+h,x:x+w]
            count,labels,stats,_=cv2.connectedComponentsWithStats(crop.astype(np.uint8),connectivity=8)
            if count<=1:
                rejected['ownership']+=1;continue
            sizes=stats[1:,cv2.CC_STAT_AREA];largest=int(sizes.max())
            if largest<area:rejected['ownership']+=1;continue
            if np.count_nonzero(sizes==largest)>1:
                # Different OpenCV image shapes may assign tied component IDs in
                # different orders. Preserve the full-size reference in this case.
                available=mask&~occupied;available=selected_component(available,available)
            else:
                available=np.zeros(mask.shape,bool);available[y:y+h,x:x+w]=labels==(int(sizes.argmax())+1)
        else:available=mask
        result=encode_mask(available);selected.append(result);native_selected.append(native_rle(result));occupied|=available
    return selected,rejected

