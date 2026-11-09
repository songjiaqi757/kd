#!/usr/bin/env python3
"""Paired legacy/corrected temporal metadata with one frozen teacher load."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'project/src'))
from rdid_mosei.benchmark import parse_sentiment_score
from rdid_mosei.metrics import sentiment_metrics
from rdid_mosei.probe import extract_thinker_last_input_state
from rdid_mosei.subsets import build_conversation
from rdid_mosei.teacher_video import read_teacher_media, video_preprocessing_audit
from train_video_adaptation import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT/'model/Qwen3-Omni-30B-A3B-Instruct')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    assert rows and {r['split'] for r in rows} == {'train'}
    assert len({r['sample_id'] for r in rows}) == len(rows)
    args.output.mkdir(parents=True,exist_ok=True)
    identity = {'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'model':str(args.model.resolve()),'subsets':['v','ta','tav'],'seed':2026,
                'scope':'train-only paired input diagnostic; no fitting or official-test evaluation'}
    cp = args.output/'config.json'
    if cp.exists() and json.loads(cp.read_text()) != identity:
        raise ValueError('diagnostic resume identity mismatch')
    atomic_json(identity,cp)
    jobs = [(n,r,s) for n,r in enumerate(rows) for s in identity['subsets']]
    pending = [(n,r,s) for n,r,s in jobs if not (args.output/f'{n:03d}_{s}.json').exists()]
    if pending:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError('paired teacher diagnostic requires two visible GPUs')
        os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'
        torch.manual_seed(2026)
        processor = Qwen3OmniMoeProcessor.from_pretrained(args.model,local_files_only=True)
        cfg = Qwen3OmniMoeConfig.from_pretrained(args.model,local_files_only=True)
        cfg.enable_audio_output = False
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            args.model,config=cfg,local_files_only=True,dtype=torch.bfloat16,
            device_map='balanced',attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
        for step,(n,row,subset) in enumerate(pending,1):
            started = time.time()
            conv = build_conversation(row,subset,prompt_version='v1')
            a,i,v,kw = read_teacher_media(conv,'sampled_fps_v2')
            prompt = processor.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
            common = dict(text=prompt,audio=a,images=i,videos=v,return_tensors='pt',padding=True,use_audio_in_video=False)
            old = processor(**common)
            new = processor(**common,**kw)
            if not torch.equal(old['input_ids'],new['input_ids']):
                raise ValueError('paired diagnostic changed text tokens')
            if v is not None and not torch.equal(old['pixel_values_videos'],new['pixel_values_videos']):
                raise ValueError('paired diagnostic changed pixels')
            record = {'sample_id':row['sample_id'],'video_id':row['video_id'],
                'subset':subset,'target_sentiment':row['sentiment'],'variants':{}}
            vectors = {}
            for variant,inputs,kwargs in [('legacy',old,{}),('corrected',new,kw)]:
                audit = video_preprocessing_audit(inputs,processor,kwargs)
                inputs = inputs.to(model.device).to(model.dtype)
                with torch.inference_mode():
                    hidden = extract_thinker_last_input_state(model,inputs)[0].float().cpu().numpy()
                if not np.isfinite(hidden).all():
                    raise ValueError('non-finite diagnostic hidden')
                vectors[variant] = hidden
                item = {'preprocessing':audit}
                if subset == 'v':
                    with torch.inference_mode():
                        generated = model.generate(**inputs,return_audio=False,thinker_max_new_tokens=32,do_sample=False)
                    raw = processor.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)[0]
                    item['output_text'] = raw
                    try:
                        item['score'] = parse_sentiment_score(raw)
                    except ValueError as exc:
                        item['score'],item['parse_error'] = None,str(exc)
                record['variants'][variant] = item
                del inputs
            left,right = vectors['legacy'],vectors['corrected']
            record['hidden_relative_l2'] = float(np.linalg.norm(right-left)/max(float(np.linalg.norm(left)),1e-12))
            record['hidden_max_abs_change'] = float(np.max(abs(right-left)))
            if subset == 'ta' and not np.array_equal(left,right):
                raise ValueError('TA negative control changed despite identical inputs')
            feature = args.output/f'{n:03d}_{subset}.npz'
            tmp = feature.with_suffix('.npz.tmp')
            with tmp.open('wb') as stream:
                np.savez(stream,**vectors)
            os.replace(tmp,feature)
            record['feature_sha256'] = hashlib.sha256(feature.read_bytes()).hexdigest()
            atomic_json(record,args.output/f'{n:03d}_{subset}.json')
            print(json.dumps({'stage':'paired_teacher','completed':len(jobs)-len(pending)+step,
                'total':len(jobs),'sample_id':row['sample_id'],'subset':subset,
                'seconds':time.time()-started,'hidden_relative_l2':record['hidden_relative_l2']}),flush=True)
            del old,new,v,a,i
            torch.cuda.empty_cache()
    records=[]
    for n,row,subset in jobs:
        record=json.loads((args.output/f'{n:03d}_{subset}.json').read_text())
        feature=args.output/f'{n:03d}_{subset}.npz'
        if hashlib.sha256(feature.read_bytes()).hexdigest()!=record['feature_sha256']:
            raise ValueError('diagnostic feature checksum mismatch')
        records.append(record)
    summary={'scope':identity['scope'],'samples':len(rows),'jobs':len(records),'direct_v':{},'hidden':{},
        'note':'A deliberately stratified train diagnostic; zero fraction is not comparable to the old benchmark500 population.'}
    vr=[r for r in records if r['subset']=='v']
    paired=[r for r in vr if all(r['variants'][v]['score'] is not None for v in ['legacy','corrected'])]
    for variant in ['legacy','corrected']:
        scores=[r['variants'][variant]['score'] for r in paired]
        summary['direct_v'][variant]={'jointly_parsed':len(scores),'parse_failures':sum(r['variants'][variant]['score'] is None for r in vr),
            'zero_fraction':float(np.mean(np.array(scores)==0)) if scores else None,
            'metrics':sentiment_metrics([r['target_sentiment'] for r in paired],scores) if scores else None}
    for subset in identity['subsets']:
        sr=[r for r in records if r['subset']==subset]
        summary['hidden'][subset]={'mean_relative_l2':float(np.mean([r['hidden_relative_l2'] for r in sr])),
            'max_abs_change':max(r['hidden_max_abs_change'] for r in sr)}
    atomic_json(summary,args.output/'summary.json')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
