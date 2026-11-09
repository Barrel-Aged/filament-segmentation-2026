"""Explicit full-supplied-training refit scope and strict CSV exchange; CPU only.

This index is for authorized competition deployment, never development scoring.
The ordinary ObservationIndex holdout guard is unchanged.
"""
import csv
from pathlib import Path
import numpy as np
from .common import read_json, sha256
from .dataset import ObservationIndex
from .metrics import decode_mask, encode_mask, validate_instances


class SubmissionRefitIndex(ObservationIndex):
    def __init__(self, manifest_path='manifests/submission-s1-data.json'):
        manifest = read_json(manifest_path)
        if manifest['purpose'] != 'authorized_full_supplied_training_refit_no_validation':
            raise ValueError('Explicit deployment-only manifest required')
        super().__init__(validation_fold=0)
        for path, digest in manifest['input_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('Deployment input changed: ' + path)
        if manifest['fit_names'] != sorted(self.images):
            raise ValueError('Refit must contain every supplied training image exactly once')
        self.train_names = manifest['fit_names']
        self.inner_names = []
        self.validation_names = []
        self.refit_manifest_sha256 = sha256(manifest_path)

    def development_record(self, record_id):
        # AnnotationCache calls this interface. Access is granted only by the
        # explicit deployment manifest above; no development index is relaxed.
        record = self.records[record_id]
        if record['file_name'] not in self.train_names:
            raise ValueError('Not a supplied refit target')
        return record


def export_csv(predictions, path, shapes):
    if set(predictions) != set(shapes):
        raise ValueError('Every test image needs an inference record, including empty images')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    rows = []
    for name in sorted(shapes):
        stem = Path(name).stem
        if '_' in stem:
            raise ValueError('Unexpected observation identifier')
        validate_instances(predictions[name], shapes[name])
        for number, rle in enumerate(predictions[name], 1):
            mask = decode_mask(rle)
            if encode_mask(mask) != rle:
                raise ValueError('Noncanonical or malformed COCO RLE')
            rows.append((f'{stem}_{number}', rle['counts']))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['filament_id', 'segmentation_rle'])
        writer.writerows(rows)
    return verify_csv(path, predictions, shapes)


def verify_csv(path, predictions, shapes):
    if set(predictions) != set(shapes):
        raise ValueError('Prediction coverage differs from test inventory')
    stems = {Path(name).stem: name for name in shapes}
    if len(stems) != len(shapes):
        raise ValueError('Duplicate observation stem')
    reconstructed = {name: [] for name in shapes}
    seen = set()
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['filament_id', 'segmentation_rle']:
            raise ValueError('Wrong submission columns')
        for row in reader:
            key = row['filament_id']
            if key in seen or None in row:
                raise ValueError('Duplicate ID or extra CSV columns')
            seen.add(key)
            stem, number = key.rsplit('_', 1)
            if stem not in stems or int(number) != len(reconstructed[stems[stem]]) + 1:
                raise ValueError('Unknown observation or nonsequential instance ID')
            name = stems[stem]
            rle = {'size': list(shapes[name]), 'counts': row['segmentation_rle']}
            mask = decode_mask(rle)
            if encode_mask(mask) != rle:
                raise ValueError('CSV RLE does not round-trip')
            reconstructed[name].append(rle)
    if reconstructed != predictions:
        raise ValueError('CSV differs from frozen predictions')
    for name, rles in reconstructed.items():
        validate_instances(rles, shapes[name])
    return {'passed': True, 'csv_sha256': sha256(path), 'images_inferred': len(shapes),
            'images_with_instances': sum(bool(v) for v in reconstructed.values()),
            'empty_images': sorted(n for n, v in reconstructed.items() if not v),
            'instance_rows': len(seen), 'rle_orientation': 'COCO column-major; exact mask round-trip',
            'disjoint_instances': True, 'invented_empty_image_predictions': False}
