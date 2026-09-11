#!/usr/bin/env python3
"""CPU-only real-media check: identical pixels, corrected teacher temporal positions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeProcessor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoePreTrainedModelForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'project/src'))
from rdid_mosei.subsets import build_conversation
from rdid_mosei.teacher_video import read_teacher_media, video_preprocessing_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/teacher_video_timing_audit_20260911.json')
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error('--limit must be positive')
    os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'
    model_path = ROOT/'model/Qwen3-Omni-30B-A3B-Instruct'
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, local_files_only=True)
    config = Qwen3OmniMoeConfig.from_pretrained(model_path, local_files_only=True)
    # This base class only supplies position-index math; no teacher weights/layers.
    position_model = Qwen3OmniMoePreTrainedModelForConditionalGeneration(config.thinker_config)
    position_model.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
    assert sum(p.numel() for p in position_model.parameters()) == 0
    manifest = ROOT/'dataset/cmu_mosei/manifests/official_train_valid_windowed.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = [r for r in rows if r['split'] == 'train'][:args.limit]
    result = {'created_at':datetime.now(timezone.utc).isoformat(),
        'scope':'CPU preprocessing and position-index computation only; first train windows; no teacher inference or test access.',
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'samples':[]}
    for row in selected:
        conversation = build_conversation(row, 'v')
        a, i, videos, kwargs = read_teacher_media(conversation, 'sampled_fps_v2')
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        base = dict(text=prompt,audio=a,images=i,videos=videos,return_tensors='pt',padding=True,use_audio_in_video=False)
        old = processor(**base)
        fixed = processor(**base, **kwargs)
        pixels_equal = torch.equal(old['pixel_values_videos'],fixed['pixel_values_videos'])
        tokens_equal = torch.equal(old['input_ids'],fixed['input_ids'])
        assert pixels_equal and tokens_equal
        positions = []
        for inputs in [old,fixed]:
            positions.append(position_model.get_rope_index(
                input_ids=inputs['input_ids'],video_grid_thw=inputs['video_grid_thw'],
                attention_mask=inputs['attention_mask'],use_audio_in_video=False,
                second_per_grids=inputs['video_second_per_grid'])[0])
        record = {'sample_id':row['sample_id'],'split':row['split'],'duration':row['duration'],
            'sampled_frames':len(videos[0]),'pixels_equal':pixels_equal,'input_ids_equal':tokens_equal,
            'legacy':video_preprocessing_audit(old,processor,{}),
            'corrected':video_preprocessing_audit(fixed,processor,kwargs),
            'temporal_position_ids_changed':not torch.equal(positions[0],positions[1]),
            'max_position_id_difference':float((positions[0]-positions[1]).abs().max())}
        result['samples'].append(record)
        print(json.dumps(record),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
