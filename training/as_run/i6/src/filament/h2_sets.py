"""H2 bounded set selection. No annotation or image-index dependency at inference."""
import itertools
import cv2
import numpy as np
from pycocotools import mask as coco
from .metrics import native_rle, decode_mask
from .g4_assembly_fast3 import encode_mask
from .g4_data import selected_component

FEATURES = ['count', 'confidence_sum', 'confidence_squared_sum', 'log_area_sum',
            'small_count', 'limb_count', 'fill_sum', 'log_aspect_sum',
            'retained_fraction_sum', 'confidence_retained_sum',
            'lost_overlap_fraction_sum', 'lost_fragment_fraction_sum',
            'original_pair_iou_sum', 'original_pair_containment_sum',
            'confidence_pair_overlap_sum', 'contrast_retained_sum',
            'duplicate_count', 'ownership_rejection_count']


def clusters(rows, threshold, minimum_area=64):
    eligible = sorted([r for r in rows if r['score'] >= threshold and r['area'] >= minimum_area],
                      key=lambda r: (-r['score'], r['index']))
    if not eligible: return [], {'confidence': sum(r['score'] < threshold for r in rows), 'area': sum(r['score'] >= threshold and r['area'] < minimum_area for r in rows)}
    rles = [native_rle(r['rle']) for r in eligible]
    overlap = coco.iou(rles, rles, [0]*len(rles)) > 0
    remaining = set(range(len(eligible))); result = []
    while remaining:
        seed = min(remaining); remaining.remove(seed); group = {seed}; stack = [seed]
        while stack:
            i = stack.pop()
            neighbors = sorted(j for j in remaining if overlap[i, j])
            remaining.difference_update(neighbors); group.update(neighbors); stack.extend(neighbors)
        result.append([eligible[i] for i in sorted(group)])
    return result, {'confidence': sum(r['score'] < threshold for r in rows), 'area': sum(r['score'] >= threshold and r['area'] < minimum_area for r in rows)}


def sequences(n, limit=6):
    """At most 121 ordered subsets: each subset in confidence/reverse order."""
    original = tuple(range(n))
    if n < 2 or n > limit: return [original]
    options = [original, ()]; seen = set(options)
    for k in range(1, n+1):
        for subset in itertools.combinations(range(n), k):
            for order in (subset, tuple(reversed(subset))):
                if order not in seen: seen.add(order); options.append(order)
    assert len(options) <= 128
    return options


class Cluster:
    def __init__(self, rows, minimum_area=64):
        self.rows = rows; self.minimum_area = minimum_area
        self.shape = tuple(rows[0]['rle']['size'])
        boxes = [coco.toBbox(native_rle(r['rle'])).astype(int) for r in rows]
        self.x = min(b[0] for b in boxes); self.y = min(b[1] for b in boxes)
        right = max(b[0]+b[2] for b in boxes); bottom = max(b[1]+b[3] for b in boxes)
        self.masks = [decode_mask(r['rle'])[self.y:bottom, self.x:right].astype(bool) for r in rows]
        self.areas = [int(m.sum()) for m in self.masks]
        assert self.areas == [r['area'] for r in rows]

    def assemble(self, sequence):
        occupied = np.zeros_like(self.masks[0]); selected = []; outputs = []
        events = []; features = np.zeros(len(FEATURES), np.float64)
        for i in sequence:
            mask = self.masks[i]; row = self.rows[i]; area = self.areas[i]
            if any(np.count_nonzero(mask & m)/np.count_nonzero(mask | m) > .5 for m in outputs):
                features[16] += 1; events.append({'index': row['index'], 'stage': 'duplicate', 'area': area}); continue
            available = mask & ~occupied; overlap_lost = area-int(available.sum())
            count, labels, stats, _ = cv2.connectedComponentsWithStats(available.astype(np.uint8), connectivity=8)
            sizes = stats[1:, cv2.CC_STAT_AREA]
            largest = int(sizes.max()) if len(sizes) else 0
            fragments = int(available.sum())-largest
            if largest < self.minimum_area:
                features[17] += 1
                events.append({'index': row['index'], 'stage': 'ownership', 'area': area,
                               'overlap_pixels': overlap_lost, 'fragment_pixels': fragments,
                               'discarded_components': len(sizes), 'discarded_pixels': int(available.sum())})
                continue
            if np.count_nonzero(sizes == largest) > 1:
                # Preserve original full-resolution OpenCV component-ID tie behavior.
                full = np.zeros(self.shape, bool); h, w = available.shape
                full[self.y:self.y+h, self.x:self.x+w] = available
                full = selected_component(full, full)
                kept = full[self.y:self.y+h, self.x:self.x+w]
            else: kept = labels == (int(sizes.argmax())+1)
            y, x = np.where(kept); width = x.max()-x.min()+1; height = y.max()-y.min()+1
            c = row['score']; fraction = largest/area
            features[:12] += [1, c, c*c, np.log1p(largest), largest < 256, row.get('rho', 0) >= .8,
                             largest/(width*height), abs(np.log(width/height)), fraction,
                             c*fraction, overlap_lost/area, fragments/area]
            features[15] += float(np.clip(row.get('features', [0]*14)[13], -10, 10))*fraction
            for j in selected:
                intersection = np.count_nonzero(mask & self.masks[j])
                iou = intersection/(area+self.areas[j]-intersection)
                features[12] += iou; features[13] += intersection/min(area, self.areas[j])
                features[14] += c*self.rows[j]['score']*iou
            occupied |= kept; selected.append(i); outputs.append(kept)
            events.append({'index': row['index'], 'stage': 'accepted', 'area': area,
                           'output_area': largest, 'overlap_pixels': overlap_lost,
                           'fragment_pixels': fragments, 'discarded_components': max(0, len(sizes)-1)})
        value = {'features': features, 'outputs': outputs, 'selected': selected, 'events': events,
                 'sequence': sequence, 'dropped': [r['index'] for i, r in enumerate(self.rows) if i not in sequence]}
        return value

    def rles(self, value):
        if 'rles' not in value:
            rles = []
            for mask in value['outputs']:
                full = np.zeros(self.shape, bool); h, w = mask.shape
                full[self.y:self.y+h, self.x:self.x+w] = mask
                rles.append(encode_mask(full))
            value['rles'] = rles
        return value['rles']


def fit_pairwise(x, y, weights, penalty=.01):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64); w = np.asarray(weights, np.float64)
    assert len(x) and np.isfinite(x).all() and np.isfinite(y).all()
    w /= w.sum(); scale = np.maximum(np.sqrt((w[:, None]*x*x).sum(0)), 1e-6)
    z = x/scale
    beta = np.linalg.solve(z.T @ (w[:, None]*z)+penalty*np.eye(z.shape[1]), z.T @ (w*y))
    return {'features': FEATURES, 'scale': scale.tolist(), 'beta': beta.tolist(), 'penalty': penalty,
            'pairs': len(y), 'weighted_mse': float(np.sum(w*(z@beta-y)**2)),
            'zero_delta_mse': float(np.sum(w*y*y))}


def select(cluster, model, limit=6, margin=.02):
    options = sequences(len(cluster.rows), limit)
    base = cluster.assemble(options[0]); best = base; best_gain = margin
    if model is not None:
        assert model['features'] == FEATURES
        scale = np.asarray(model['scale']); beta = np.asarray(model['beta'])
        for sequence in options[1:]:
            value = cluster.assemble(sequence)
            gain = float(((value['features']-base['features'])/scale)@beta)
            if gain > best_gain+1e-12: best = value; best_gain = gain
    return best, base, len(options), best_gain if best is not base else 0.


def decode(rows, threshold, model=None, limit=6, margin=.02):
    groups, filtering = clusters(rows, threshold); predictions = []; ledger = []
    for members in groups:
        cluster = Cluster(members); chosen, base, width, gain = select(cluster, model, limit, margin)
        predictions.extend(cluster.rles(chosen))
        ledger.append({'members': [r['index'] for r in members], 'size': len(members),
                       'overflow': len(members)>limit, 'alternatives': width, 'predicted_gain': gain,
                       'changed': chosen is not base, 'order': [members[i]['index'] for i in chosen['sequence']],
                       'events': chosen['events'], 'dropped': chosen['dropped']})
    return predictions, {'filtering': filtering, 'clusters': ledger}
