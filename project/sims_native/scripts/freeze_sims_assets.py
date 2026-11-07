#!/usr/bin/env python3
"""Freeze dataset-specific teacher targets; never consume official-test samples."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from sims_native_common import METRIC_PROTOCOL

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'project/sims_native/scripts'
COUNTS = {'sims': (1368, 456), 'sims_v2': (2722, 647)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=COUNTS, required=True)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    data = ROOT / 'dataset' / ('ch_sims' if args.dataset == 'sims' else 'ch_sims_v2')
    base, assets = args.base / 'base_assets', args.base / 'assets'
    if not (base / 'protocol.json').exists():
        subprocess.run([sys.executable, str(SCRIPTS / 'prepare_sims_protocol.py'),
                        '--dataset', args.dataset, '--manifest', str(data / 'manifests/official_train_valid_windowed.jsonl'),
                        '--features', str(args.base / 'teacher_features'), '--probes-root', str(args.base / 'probes'),
                        '--output', str(base), '--expected-train', str(COUNTS[args.dataset][0]),
                        '--expected-valid', str(COUNTS[args.dataset][1])], check=True)
    if not (assets / 'protocol.json').exists():
        subprocess.run([sys.executable, str(SCRIPTS / 'prepare_main_table_assets.py'),
                        '--base', str(base / 'protocol.json'), '--output', str(assets)], check=True)
    protocol = json.loads((assets / 'protocol.json').read_text())
    assert protocol['dataset'] == args.dataset
    assert protocol['classification_classes'] == 5 and protocol['sentiment_scale'] == 1.0
    for path, digest in protocol['input_sha256'].items():
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert actual == digest, 'Frozen asset mismatch: ' + path
    (args.base / 'native_metric_protocol.json').write_text(json.dumps(METRIC_PROTOCOL, indent=2) + '\n')


if __name__ == '__main__':
    main()
