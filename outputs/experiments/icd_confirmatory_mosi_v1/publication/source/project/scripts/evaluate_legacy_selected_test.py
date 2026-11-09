#!/usr/bin/env python3
"""Evaluate a validation-selected historical student without teacher targets.

Supports frozen-feature cores, T/A LoRA with cached V, video adaptation, and
fixed-student main-table checkpoints. Does not modify source runs or weights.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'project/src'), str(ROOT/'project/scripts')]
from summarize_all_dual_checkpoints import select, metrics, consistent
from train_student_baseline import aggregate_windows, seed_everything, FeatureDataset, collate_features
from train_tav_distillation import sha256
from evaluate_tav_test import load_test_rows, atomic_json
from rdid_mosei.metrics import sentiment_metrics


def localize(value):
    if isinstance(value, dict):
        return {k:localize(v) for k,v in value.items()}
    if isinstance(value, list):
        return [localize(x) for x in value]
    if isinstance(value, str) and value.startswith('/ai/sjq/kd/'):
        return str(ROOT/value[len('/ai/sjq/kd/'):])
    return value


def load_selected(run, checkpoint, policy):
    history=json.loads((run/'history.json').read_text())
    row=select(history,policy)
    if row is None:
        raise ValueError('No validation selection')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('epoch') != row['epoch']:
        raise ValueError(f'Checkpoint epoch {saved.get("epoch")} != selected epoch {row["epoch"]}')
    if 'valid_metrics' in saved:
        if not consistent(saved['valid_metrics'],metrics(row)):
            raise ValueError('Checkpoint validation metrics differ from selected history')
    else:
        # Early video-adaptation writers stored only model + epoch in best.pt.
        report=json.loads((run/'report.json').read_text())
        if checkpoint.resolve() != (run/'best.pt').resolve() or report.get('best_epoch')!=row['epoch'] or not consistent(report.get('valid_metrics',{}),metrics(row)):
            raise ValueError('Legacy epoch-only checkpoint lacks matching original best report')
    return saved,row


class InputCollator:
    def __init__(self, tokenizer, audio, family, max_tokens=256):
        self.tokenizer,self.audio,self.family,self.max_tokens=tokenizer,audio,family,max_tokens

    def __call__(self, items):
        rows=[x['row'] for x in items]
        text=self.tokenizer([r['text'] for r in rows],padding=True,truncation=True,max_length=self.max_tokens,return_tensors='pt')
        audio=self.audio([x['waveform'] if self.family=='lora' else x['wave'] for x in items],sampling_rate=16000,padding=True,return_attention_mask=True,return_tensors='pt')
        inputs=dict(input_ids=text['input_ids'],text_attention_mask=text['attention_mask'],input_values=audio['input_values'],audio_attention_mask=audio.get('attention_mask'))
        if self.family=='lora':
            videos=[x['video'] for x in items];lengths=torch.tensor([v.shape[0] for v in videos])
            video=pad_sequence(videos,batch_first=True)
            inputs.update(video_hidden_states=video,video_attention_mask=torch.arange(video.shape[1])[None,:]<lengths[:,None])
        else:
            inputs['pixel_values']=None if items[0]['pixels'] is None else torch.stack([x['pixels'] for x in items])
        return dict(inputs=inputs,rows=rows)


def build(family,config,state,device):
    from rdid_mosei.student import CachedStudentCore,LoRATextAudioCachedVideoStudent
    if family=='cached':
        dims={m:state[f'pools.{m}.input_projection.weight'].shape[1] for m in 'tav'}
        model=CachedStudentCore(text_hidden_size=dims['t'],audio_hidden_size=dims['a'],video_hidden_size=dims['v'])
        if 'hidden_projection.weight' in state:
            shape=state['hidden_projection.weight'].shape
            model.hidden_projection=torch.nn.Linear(shape[1],shape[0])
        model.load_state_dict(state,strict=True)
        return model.to(device)
    if family=='lora':
        from transformers import AutoModel
        dtype=torch.bfloat16 if device.type=='cuda' else torch.float32
        text=AutoModel.from_pretrained(config['text_model'],local_files_only=True,dtype=dtype)
        audio=AutoModel.from_pretrained(config['audio_model'],local_files_only=True,dtype=dtype)
        text.config.use_cache=False
        model=LoRATextAudioCachedVideoStudent(text,audio,video_hidden_size=state['pools.v.input_projection.weight'].shape[1],
                lora_rank=config['lora_rank'],lora_alpha=config['lora_alpha'],lora_dropout=config['lora_dropout'])
        if config.get('freeze_text_audio'):
            model.text_encoder.requires_grad_(False);model.audio_encoder.requires_grad_(False)
        expected={k for k,p in model.named_parameters() if p.requires_grad}
        if set(state)!=expected:
            raise ValueError(f'LoRA checkpoint keys differ: missing={expected-set(state)}, extra={set(state)-expected}')
        model.load_state_dict(state,strict=False)
        return model.to(device)
    args=SimpleNamespace(**config)
    if family=='video':
        from train_video_adaptation import build_model,load_trainable
        model=build_model(args,device);load_trainable(model,state)
    elif family=='main':
        from train_main_table_kd import build_model,load_checkpoint_state
        model=build_model(args,config['frozen_assets'],device);load_checkpoint_state(model,state)
    else:
        raise ValueError(family)
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--test-manifest',type=Path,required=True)
    p.add_argument('--checkpoint-selection',choices=('valid_mae','valid_acc2_nonzero'),required=True)
    p.add_argument('--family',choices=('cached','lora','video','main'),required=True)
    p.add_argument('--features',type=Path);p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--allow-official-test',action='store_true');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if not args.allow_official_test:p.error('Explicit official-test authorization is required')
    run=args.run.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    if not (run/'report.json').exists():raise ValueError('Training report missing')
    if (run/'status.json').exists() and json.loads((run/'status.json').read_text()).get('status')!='complete':
        raise ValueError('Training is not complete')
    config=json.loads((run/'run_config.json').read_text()) if (run/'run_config.json').exists() else {}
    saved,row=load_selected(run,args.checkpoint,args.checkpoint_selection)
    rows,parents=load_test_rows(args.test_manifest)
    protocol=dict(schema='historical-selected-test-v1',source_run=str(run),checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=sha256(args.checkpoint),selected_epoch=row['epoch'],checkpoint_selection=args.checkpoint_selection,
                  selected_valid_metrics=metrics(row),manifest=str(args.test_manifest.resolve()),manifest_sha256=sha256(args.test_manifest),
                  family=args.family,features=str(args.features) if args.features else None,
                  source_run_config_sha256=sha256(run/'run_config.json') if config else None,
                  official_test_evaluated=True,test_use_policy='user-authorized retrospective report; no test-based checkpoint selection')
    if args.dry_run:
        print(json.dumps(protocol));return
    config=localize(config)
    device=torch.device(args.device)
    seed_everything(int(config.get('seed',json.loads((run/'report.json').read_text()).get('seed',13))),True)
    if args.family in ('cached','lora'):
        fc=json.loads((args.features/'run_config.json').read_text())
        if fc['status']!='complete' or fc['manifest_sha256']!=sha256(args.test_manifest):
            raise ValueError('Test feature cache incomplete or manifest mismatch')
        index=[json.loads(x) for x in (args.features/'index.jsonl').read_text().splitlines() if x]
        byid={r['sample_id']:r for r in index}
        if len(byid)!=len(rows) or set(byid)!={r['sample_id'] for r in rows}:raise ValueError('Test cache coverage mismatch')
        for r in rows:
            cached=byid[r['sample_id']]
            if cached['split']!='test' or cached['sentiment']!=r['sentiment']:raise ValueError('Cache metadata mismatch')
            r['feature_path']=cached['feature_path'];r['sample_weight']=r['aggregation_weight']
        protocol['feature_config_sha256']=sha256(args.features/'run_config.json')
    output.mkdir(parents=True)
    atomic_json(protocol,output/'run_config.json');atomic_json({'status':'loading'},output/'status.json')
    model=build(args.family,config,saved['model'],device);del saved
    if args.family=='cached':
        dataset=FeatureDataset(rows);collator=collate_features
    else:
        from transformers import AutoTokenizer,AutoFeatureExtractor,AutoImageProcessor
        tokenizer=AutoTokenizer.from_pretrained(config['text_model'],local_files_only=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        audio=AutoFeatureExtractor.from_pretrained(config['audio_model'],local_files_only=True)
        if args.family=='lora':
            from train_student_lora import OnlineDataset
            dataset=OnlineDataset(rows)
        else:
            from train_video_adaptation import VideoDataset
            processor=AutoImageProcessor.from_pretrained(config['video_model'],local_files_only=True,use_fast=False)
            dataset=VideoDataset(rows,processor,mode=config.get('mode','video_lora'))
        collator=InputCollator(tokenizer,audio,args.family,config.get('max_text_tokens',256))
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collator,pin_memory=device.type=='cuda')
    model.eval();records=[];started=time.perf_counter()
    with torch.inference_mode():
        for step,batch in enumerate(loader,1):
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                if args.family=='cached':
                    pred=model({k:v.to(device) for k,v in batch['hidden'].items()},{k:v.to(device) for k,v in batch['masks'].items()},subsets=('tav',))['tav']
                else:
                    inputs={k:v.to(device) if v is not None else None for k,v in batch['inputs'].items()}
                    pred=model(**inputs,subsets=('tav',))['tav'] if args.family=='lora' else model(**inputs)
            for r,score,logits in zip(batch['rows'],pred['regression'].float().cpu().tolist(),pred['classification_logits'].float().cpu().tolist()):
                records.append(dict(sample_id=r['sample_id'],parent_sample_id=r['parent_sample_id'],split='test',
                   target_sentiment=r['sentiment'],class_7_index=r['class_7_index'],aggregation_weight=r['aggregation_weight'],prediction=score,classification_logits=logits))
            if step==1 or step%50==0 or step==len(loader):
                atomic_json(dict(status='evaluating',step=step,steps=len(loader)),output/'status.json')
    utterances=aggregate_windows(records)
    if len(utterances)!=len(parents):raise ValueError('Test utterance coverage mismatch')
    video_ids={r['parent_sample_id']:r['video_id'] for r in rows}
    for r in utterances:r['video_id']=video_ids[r['parent_sample_id']]
    result=sentiment_metrics([r['target_sentiment'] for r in utterances],[r['prediction'] for r in utterances])
    (output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in utterances))
    report={**protocol,'metrics':result,'method':run.name,'seed':config.get('seed',json.loads((run/'report.json').read_text()).get('seed')),'elapsed_seconds':time.perf_counter()-started}
    atomic_json(report,output/'report.json');atomic_json({'status':'complete'},output/'status.json')
    print(json.dumps({'output':str(output),'metrics':result}),flush=True)


if __name__=='__main__':main()
