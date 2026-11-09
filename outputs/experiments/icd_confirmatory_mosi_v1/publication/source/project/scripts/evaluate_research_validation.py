#!/usr/bin/env python3
"""Check retained validation predictions, or infer them if no matching cache exists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'project/scripts'), str(ROOT/'project/src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--registry', type=Path)
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from evaluate_main_table_test import OfficialTestCollator
    from train_main_table_kd import build_model, evaluate, load_checkpoint_state
    from train_tav_distillation import VideoDataset, sha256, atomic_json
    from rdid_mosei.decision_icd import error_strata
    from rdid_mosei.metrics import sentiment_metrics

    run = args.run.resolve()
    if json.loads((run/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Training must be complete before retained-checkpoint evaluation')
    config = json.loads((run/'run_config.json').read_text())
    report = json.loads((run/'report.json').read_text())
    manifest = Path(config['manifest'])
    all_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not {row['split'] for row in all_rows} <= {'train', 'valid'}:
        raise ValueError('Validation evaluation refuses test manifests')
    rows = [row for row in all_rows if row['split'] == 'valid']
    expected = {row['parent_sample_id'] for row in rows}
    if not expected:
        raise ValueError('No validation samples')
    output = run/'validation_postrun'
    output.mkdir(exist_ok=True)
    model = loader = None
    device = torch.device(args.device)
    summaries = {}
    for policy in ('valid_mae', 'valid_acc2_nonzero'):
        checkpoint = run/f'best_{policy}.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['checkpoint_selection'] != policy or saved['epoch'] != report['retained_best_checkpoints'][policy]['epoch']:
            raise ValueError('Retained checkpoint identity mismatch')
        cached = run/'validation'/policy
        predictions = None
        if (cached/'report.json').is_file() and (cached/'predictions.jsonl').is_file():
            metadata = json.loads((cached/'report.json').read_text())
            if (metadata['selected_epoch'] == saved['epoch'] and metadata['metrics'] == saved['valid_metrics']
                    and metadata.get('official_test_evaluated') is False):
                predictions = [json.loads(line) for line in (cached/'predictions.jsonl').read_text().splitlines()]
        reused_cache = predictions is not None
        if predictions is None:
            if model is None:
                from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer
                model_args = SimpleNamespace(**config)
                for key in ('text_model', 'audio_model', 'video_model', 'assets'):
                    setattr(model_args, key, Path(getattr(model_args, key)))
                model = build_model(model_args, config['frozen_assets'], device)
                tokenizer = AutoTokenizer.from_pretrained(model_args.text_model, local_files_only=True)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                audio = AutoFeatureExtractor.from_pretrained(model_args.audio_model, local_files_only=True)
                video = AutoImageProcessor.from_pretrained(model_args.video_model, local_files_only=True, use_fast=False)
                loader = DataLoader(VideoDataset(rows, video, mode='video_lora'), batch_size=8,
                    shuffle=False, num_workers=2, pin_memory=device.type == 'cuda',
                    collate_fn=OfficialTestCollator(tokenizer, audio))
            load_checkpoint_state(model, saved['model'])
            with torch.inference_mode():
                _, predictions = evaluate(model, loader, device)
        if len(predictions) != len(expected) or {r['parent_sample_id'] for r in predictions} != expected:
            raise ValueError('Validation prediction coverage mismatch')
        strata = error_strata(predictions)  # Also rejects test predictions and nonfinite values.
        metrics = sentiment_metrics([r['target_sentiment'] for r in predictions], [r['prediction'] for r in predictions])
        directory = output/policy
        directory.mkdir(exist_ok=True)
        temporary = directory/'predictions.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(row)+'\n' for row in predictions))
        temporary.replace(directory/'predictions.jsonl')
        item = dict(method=config.get('experiment_method', config['method']), seed=config['seed'],
                    selected_epoch=saved['epoch'], checkpoint_selection=policy,
                    checkpoint_sha256=sha256(checkpoint), metrics=metrics,
                    training_recorded_metrics=saved['valid_metrics'], reused_matching_cache=reused_cache,
                    official_test_evaluated=False, error_strata=strata,
                    source_config_sha256=sha256(run/'run_config.json'), manifest_sha256=sha256(manifest))
        atomic_json(item, directory/'report.json')
        summaries[policy] = item
        del saved
    atomic_json(summaries, output/'report.json')
    atomic_json({'status': 'complete', 'official_test_evaluated': False}, output/'status.json')
    if args.registry:
        from summarize_mosi_research import summarize
        summarize(args.registry)
    print(json.dumps({policy: r['metrics'] for policy, r in summaries.items()}), flush=True)


if __name__ == '__main__':
    main()
