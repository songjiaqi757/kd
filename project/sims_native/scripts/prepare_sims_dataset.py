#!/usr/bin/env python3
"""Prepare every officially annotated SIMS clip without changing its split."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import subprocess
import sys

from prepare_mosi_dataset import prepare_media, duration_bucket, write_jsonl, sha256_file
from sims_native_common import five_class_index

ROOT = Path(__file__).resolve().parents[3]
EXPECTED = {'sims': {'train': 1368, 'valid': 456, 'test': 457},
            'sims_v2': {'train': 2722, 'valid': 647, 'test': 1034}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=EXPECTED, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    source = ROOT / 'dataset' / ('ch_sims_source' if args.dataset == 'sims' else 'ch_sims_v2_source/supervised')
    output = ROOT / 'dataset' / ('ch_sims' if args.dataset == 'sims' else 'ch_sims_v2')
    label = source / ('label.csv' if args.dataset == 'sims' else 'meta.csv')
    with label.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    if rows[0][0] == 'video_id':
        rows = rows[1:]
    assert Counter(r[8] for r in rows) == EXPECTED[args.dataset]
    assert len({(r[0], r[1]) for r in rows}) == len(rows), 'Duplicate official clip ID'

    def prepare(row):
        video, clip, text = row[:3]
        stem = video + '__' + clip
        raw = source / 'Raw' / video / (clip + '.mp4')
        silent = output / 'media/video_silent' / (stem + '.mp4')
        audio = output / 'media/audio_16k_mono' / (stem + '.wav')
        duration = prepare_media(raw, silent, audio, False)
        native = float(row[3])
        assert -1 <= native <= 1 and duration > 0
        value = five_class_index(native)
        return {
            'manifest_version': 'sims-official-split-native-media-v1',
            'dataset': args.dataset, 'sample_id': f'{video}[{clip}]',
            'video_id': video, 'clip_index': int(clip), 'source_clip_id': clip,
            'split': row[8], 'text': text.strip(), 'sentiment': native,
            'sentiment_native': native, 'sentiment_scale': 1.0,
            'class_5_value': value - 2, 'class_5_index': value,
            'sentiment_bucket': 'negative' if native < 0 else 'positive' if native > 0 else 'neutral',
            'duration_bucket': duration_bucket(duration), 'duration': duration,
            'official_start': 0.0, 'official_end': duration,
            'source_video_path': str(raw), 'silent_video_path': str(silent),
            'audio_segment_path': str(audio), 'annotation': row[7],
            'unimodal_labels_native': dict(zip(('text', 'audio', 'video'), map(float, row[4:7]))),
            'source_policy': 'pinned_HF_mirror_official_CSV_clip_splits_preserved', 'status': 'valid',
        }

    prepared, errors = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare, row): i for i, row in enumerate(rows)}
        for done, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            try:
                prepared[i] = future.result()
            except Exception as exc:
                errors.append({'row': i, 'clip': rows[i][:2], 'error': repr(exc)})
            if done % 200 == 0 or done == len(rows):
                print(json.dumps({'dataset': args.dataset, 'prepared': done, 'total': len(rows), 'errors': len(errors)}), flush=True)
    reports = output / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    (reports / 'preparation_errors.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2))
    assert not errors, f'{len(errors)} clips failed; original files preserved'
    values = [prepared[i] for i in range(len(rows))]
    manifests = output / 'manifests'
    write_jsonl(manifests / 'all.jsonl', values)
    for split in EXPECTED[args.dataset]:
        write_jsonl(manifests / (split + '.jsonl'), [r for r in values if r['split'] == split])
    overlap = {}
    for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        overlap[a + '_' + b] = sorted({r['video_id'] for r in values if r['split'] == a} &
                                     {r['video_id'] for r in values if r['split'] == b})
    inventory = {'dataset': args.dataset, 'source_csv': str(label), 'source_csv_sha256': sha256_file(label),
                 'split_counts': EXPECTED[args.dataset], 'source_video_overlap_by_official_split': overlap,
                 'split_policy': 'official_clip_splits_unchanged; source_video_overlap_recorded_not_resplit',
                 'label_scale': 1.0, 'rows': len(values)}
    (reports / 'source_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')
    scripts = ROOT / 'project/sims_native/scripts'
    subprocess.run([sys.executable, str(scripts / 'audit_prepared_media.py'), '--manifest', str(manifests / 'all.jsonl'),
                    '--output', str(reports / 'full_media_audit.json')], check=True)
    for name, splits in [('official_train_valid_windowed', ['train', 'valid']), ('official_test_windowed', ['test'])]:
        subprocess.run([sys.executable, str(scripts / 'build_windowed_manifest.py'), '--manifest', str(manifests / 'all.jsonl'),
                        '--output-manifest', str(manifests / (name + '.jsonl')),
                        '--media-dir', str(output / 'derived/windows'), '--splits', *splits], check=True)
    (reports / 'status.json').write_text(json.dumps({'status': 'complete', **inventory}, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
