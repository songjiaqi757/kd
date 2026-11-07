from pathlib import Path
from collections import defaultdict
import hashlib
import importlib
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'project/sims_native/src'), str(ROOT / 'project/src')]
GPU = 'GPU-ac40f187-8891-00f2-3999-959bbdcf952f'


def atomic(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def domain(label_space):
    if label_space not in ('msa', 'sims'):
        raise ValueError('Unknown label space')
    package = 'rdid_sims' if label_space == 'sims' else 'rdid_mosei'
    return (importlib.import_module(package + '.probe').TeacherProbe,
            importlib.import_module(package + '.metrics').sentiment_metrics)


def load_probe(checkpoint_path, label_space):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert checkpoint['seed'] == 2026
    assert checkpoint['classes'] == (5 if label_space == 'sims' else 7)
    Probe, _ = domain(label_space)
    model = Probe(input_size=checkpoint['input_size'], hidden_size=checkpoint['hidden_size'],
                  classes=checkpoint['classes'], dropout=checkpoint['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model, checkpoint


def aggregate(rows, scores, logits):
    assert len(rows) == len(scores) == len(logits)
    assert len({row['sample_id'] for row in rows}) == len(rows), 'Duplicate input window'
    grouped = defaultdict(list)
    for row, score, logit in zip(rows, scores, logits):
        assert row['subset'] == 'tav' and float(row['aggregation_weight']) > 0
        grouped[row['parent_sample_id']].append((row, float(score), np.asarray(logit, dtype=np.float64)))
    output = []
    for parent, items in sorted(grouped.items()):
        first = items[0][0]
        assert len({(row['split'], float(row['target_sentiment'])) for row, _, _ in items}) == 1
        weight = sum(float(row['aggregation_weight']) for row, _, _ in items)
        output.append({'parent_sample_id': parent, 'split': first['split'], 'subset': 'tav',
            'target_sentiment': float(first['target_sentiment']),
            'probe_score': sum(score * float(row['aggregation_weight']) for row, score, _ in items) / weight,
            'classification_logits': (sum(logit * float(row['aggregation_weight']) for row, _, logit in items) / weight).tolist(),
            'window_count': len(items), 'aggregation_weight_sum': weight})
    return output


def apply_probe(feature_folder, checkpoint_path, label_space, split):
    feature_folder = Path(feature_folder)
    rows = [json.loads(line) for line in (feature_folder / 'index.jsonl').read_text().splitlines() if line]
    completed = np.load(feature_folder / 'completed.npy', mmap_mode='r')
    hidden = np.load(feature_folder / 'features.npy', mmap_mode='r')
    indices = [i for i, row in enumerate(rows) if row['subset'] == 'tav' and row['split'] == split]
    assert indices and completed[indices].all()
    model, checkpoint = load_probe(checkpoint_path, label_space)
    assert hidden.shape[1] == checkpoint['input_size']
    scores, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            features = np.array(hidden[indices[start:start + 128]], dtype=np.float32, copy=True)
            assert np.isfinite(features).all()
            result = model(torch.from_numpy(features))
            scores.extend(result['regression'].numpy().tolist())
            logits.extend(result['classification_logits'].numpy().tolist())
    bound = 1 if label_space == 'sims' else 3
    assert all(math.isfinite(value) and -bound <= value <= bound for value in scores)
    output = aggregate([rows[i] for i in indices], scores, logits)
    _, metric = domain(label_space)
    metrics = metric([row['target_sentiment'] for row in output], [row['probe_score'] for row in output])
    return output, metrics, checkpoint


def verify_inputs(spec):
    assert spec['official_test_authorized'] is True and spec['probe_seed'] == 2026
    for path, digest in spec['input_sha256'].items():
        assert sha256(path) == digest, 'Locked teacher-test input changed: ' + path
    for path, digest in spec['source_sha256'].items():
        assert sha256(path) == digest, 'Locked teacher-test source changed: ' + path
    for path, expected in spec.get('model_file_identity', {}).items():
        info = Path(path).stat()
        assert {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns} == expected, 'Teacher weight file changed: ' + path
