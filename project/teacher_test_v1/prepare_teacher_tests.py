"""Prepare a locked, single-Probe test extension; do not start GPU jobs."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json
import os
import sys
import numpy as np
import torch

from teacher_test_common import ROOT, GPU, atomic, apply_probe, sha256, domain

CODE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/experiments/teacher_test_v1'
QUEUE = ROOT / 'outputs/experiments/sims_native_v1/queue'
PYTHON = sys.executable
DATASETS = [
    ('mosi', 'cmu_mosi', ROOT / 'outputs/experiments/main_table_v1/mosi', 686, 'msa'),
    ('mosei', 'cmu_mosei', ROOT / 'outputs/experiments/video_source_v2', 4659, 'msa'),
    ('sims', 'ch_sims', ROOT / 'outputs/experiments/sims_native_v1/sims', 457, 'sims'),
    ('sims_v2', 'ch_sims_v2', ROOT / 'outputs/experiments/sims_native_v1/sims_v2', 1034, 'sims'),
]


def main():
    assert not (OUTPUT / 'plan.json').exists(), 'Teacher-test plan already frozen'
    torch.set_num_threads(4)
    old = json.loads((QUEUE / 'plan.json').read_text())
    assert old['jobs'][-1]['id'] == 'sims_summary'
    assert len(old['jobs']) == 145
    runtime = ['teacher_test_common.py', 'extract_teacher_test_features.py', 'run_teacher_test.py', 'summarize_teacher_tests.py']
    sources = {str(CODE / name): sha256(CODE / name) for name in runtime}
    for package, directory in [('rdid_mosei', ROOT/'project/src'), ('rdid_sims', ROOT/'project/sims_native/src')]:
        for name in ['probe.py', 'metrics.py', 'subsets.py', 'teacher_video.py', 'benchmark.py']:
            p = directory / package / name; sources[str(p)] = sha256(p)
    specs, validation, jobs = {}, {}, []
    for dataset, folder, base, count, label_space in DATASETS:
        checkpoint = base / 'probes/seed2026/teacher_probe.pt'
        report_path = base / 'probes/seed2026/report.json'
        report = json.loads(report_path.read_text())
        assert report['seed'] == 2026 and not report['official_test_evaluated']
        features = Path(report['features'])
        config_path = features / 'run_config.json'
        config = json.loads(config_path.read_text())
        assert config['prompt_version'] == 'v1' and config['video_timing_policy'] == 'sampled_fps_v2'
        assert Path(config['model']).resolve() == (ROOT / 'model/Qwen3-Omni-30B-A3B-Instruct').resolve()
        manifest = ROOT / 'dataset' / folder / 'manifests/official_test_windowed.jsonl'
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        assert {row['split'] for row in rows} == {'test'}
        assert len({row['sample_id'] for row in rows}) == len(rows)
        parents = {row['parent_sample_id'] for row in rows}; assert len(parents) == count
        training_rows = [json.loads(line) for line in Path(config['manifest']).read_text().splitlines() if line]
        assert not parents & {row['parent_sample_id'] for row in training_rows}, 'Train/valid and test clip overlap'
        for row in rows:
            assert row['aggregation_weight'] > 0
            for key in ['silent_video_path', 'audio_segment_path']: assert Path(row[key]).is_file()
        # End-to-end CPU reproduction of the already frozen Probe validation output.
        predictions, metrics, loaded = apply_probe(features, checkpoint, label_space, 'valid')
        expected = {row['parent_sample_id']: row for row in [json.loads(line) for line in (base/'probes/seed2026/predictions.jsonl').read_text().splitlines()] if row['split']=='valid' and row['subset']=='tav'}
        assert {row['parent_sample_id'] for row in predictions} == set(expected)
        delta = max(abs(row['probe_score']-expected[row['parent_sample_id']]['probe_score']) for row in predictions)
        assert delta < 1e-5, (dataset, delta)
        assert abs(metrics['mae']-report['metrics']['valid']['tav']['mae']) < 1e-5
        assert loaded['best_epoch'] == report['best_epoch']
        validation[dataset] = {'status':'passed','validation_parents':len(predictions),'maximum_prediction_difference':delta,
            'test_parents':count,'test_windows':len(rows),'probe_seed':2026,'range':[-1,1] if label_space=='sims' else [-3,3]}
        model = Path(config['model']).resolve()
        frozen_inputs = {str(path):sha256(path) for path in [checkpoint,report_path,config_path,manifest,*sorted(model.glob('*.json'))]}
        model_identity = {str(path): {'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns} for path in sorted(model.glob('*.safetensors'))}
        assert model_identity, 'Teacher weight shards missing'
        output = OUTPUT / dataset
        spec = {'dataset':dataset,'label_space':label_space,'output':str(output),'model':str(Path(config['model']).resolve()),
            'probe_seed':2026,'probe_checkpoint':str(checkpoint),'probe_report':str(report_path),
            'training_feature_config':str(config_path),'test_manifest':str(manifest),
            'expected_test_parents':count,'expected_test_windows':len(rows),'official_test_authorized':True,
            'input_sha256':frozen_inputs,'source_sha256':sources,'model_file_identity':model_identity,
            'after_completion_markers':[j['completion_marker'] for j in old['jobs']]}
        path = OUTPUT / 'specs' / (dataset+'.json');atomic(spec,path);specs[dataset]=spec
        jobs.append({'id':dataset+'_teacher30b_test_seed2026','kind':'teacher','run':str(output),
            'memory_mib':75000,'disk_mib':1000,'priority':20,
            'dependencies':['sims_summary'] if not jobs else [jobs[-1]['id']],
            'completion_marker':str(output/'status.json'),
            'command':[PYTHON,'-u',str(CODE/'run_teacher_test.py'),'--spec',str(path)]})
    # Meaningful aggregation check: overlapping windows count as one parent and retain weights.
    from teacher_test_common import aggregate
    rows=[{'sample_id':'x::w0','parent_sample_id':'x','subset':'tav','split':'test','target_sentiment':.2,'aggregation_weight':2},
          {'sample_id':'x::w1','parent_sample_id':'x','subset':'tav','split':'test','target_sentiment':.2,'aggregation_weight':1}]
    result=aggregate(rows,[.3,.0],[[1,2],[4,5]])
    assert len(result)==1 and abs(result[0]['probe_score']-.2)<1e-8 and result[0]['classification_logits']==[2,3]
    snapshot={'schema':'single_probe_four_dataset_teacher_test_v1','created_at':datetime.now(timezone.utc).isoformat(),
        'output':str(OUTPUT),'model':'Qwen3-Omni-30B-A3B-Instruct','probe_seed':2026,'physical_gpu':1,'gpu_uuid':GPU,
        'after_student_jobs':145,'datasets':specs,'source_sha256':sources,
        'scope':'Frozen 30B+existing seed2026 Probe; complete test sets; TAV only; no fitting or test-based selection'}
    atomic(snapshot,OUTPUT/'plan.json');atomic(validation,OUTPUT/'preflight_verification.json')
    jobs.append({'id':'teacher30b_four_dataset_test_summary','kind':'summary','run':str(OUTPUT),
        'memory_mib':1,'disk_mib':20,'priority':20,'dependencies':[j['id'] for j in jobs],
        'completion_marker':str(OUTPUT/'summary_status.json'),
        'command':[PYTHON,'-u',str(CODE/'summarize_teacher_tests.py'),'--plan',str(OUTPUT/'plan.json')]})
    new=json.loads(json.dumps(old));new['jobs'].extend(jobs);new['source_sha256'].update(sources)
    new['authorized_post_campaign_extension']={'requested':'After all existing experiments finish, test the teacher on four complete test splits, one Probe seed',
        'teacher_test_plan':str(OUTPUT/'plan.json'),'original_jobs':145,'added_jobs':5}
    sys.path.insert(0,str(ROOT/'project/sims_native/scripts'))
    from run_sims_gpu_queue import validate_plan, ready_jobs
    validate_plan(new)
    assert new['jobs'][:145]==old['jobs']
    fake={j['id']:{'state':'pending'} for j in new['jobs']}
    assert not any(j['id'] in {x['id'] for x in jobs} for j in ready_jobs(new['jobs'],fake))
    fake.update({j['id']:{'state':'complete'} for j in old['jobs']})
    assert [j['id'] for j in ready_jobs(new['jobs'],fake)]==[jobs[0]['id']]
    atomic(new,QUEUE/'plan.teacher_test_candidate.json')
    print(json.dumps({'status':'prepared','verification':validation,'new_jobs':len(jobs),'candidate':str(QUEUE/'plan.teacher_test_candidate.json')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
