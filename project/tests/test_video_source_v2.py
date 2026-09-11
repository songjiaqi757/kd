import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_video_source_v2 as queue
import train_video_adaptation_v2 as trainer


def test_diagnostic_is_stratified_train_only_and_source_disjoint():
    rows=[]
    for sign in [-1,0,1]:
        for n in range(30):
            for index in [0,1]:
                rows.append({'split':'train','window_count':1,'sentiment':sign,
                    'sample_id':f'{sign}_{n}_{index}','video_id':f'{sign}_{n}'})
    rows.append({'split':'valid','window_count':1,'sentiment':1,'sample_id':'forbidden','video_id':'forbidden'})
    selected=queue.select_diagnostic(rows)
    assert len(selected)==64 and len({r['video_id'] for r in selected})==64
    assert [sum(r['sentiment']==s for r in selected) for s in [-1,0,1]]==[22,20,22]
    assert all(r['split']=='train' for r in selected)
    assert selected==queue.select_diagnostic(rows)


def test_gpu_query_failure_is_not_idle(monkeypatch):
    monkeypatch.setattr(queue.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stderr='NVML failure'))
    assert not queue.gpu_idle(0)[0]


def test_ta_control_can_use_identical_tav_teacher(tmp_path):
    rows=[{'sample_id':s,'parent_sample_id':s,'video_id':s,'split':s,'sentiment':1.,
        'class_7_index':4,'aggregation_weight':1.,'duration':1.} for s in ['train','valid']]
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    targets=[]
    for r in rows:
        for subset,score in [('ta',.2),('tav',.8)]:
            targets.append({'parent_sample_id':r['parent_sample_id'],'split':r['split'],
                'target_sentiment':1.,'class_7_index':4,'subset':subset,'probe_score':score,'classification_logits':[0.]*7})
    target=tmp_path/'predictions.jsonl';target.write_text(''.join(json.dumps(r)+'\n' for r in targets))
    tr,va=trainer.load_rows(SimpleNamespace(manifest=manifest,teacher_targets=target,mode='ta_only',teacher_subset='tav',limit_per_split=None))
    assert tr[0]['teacher_score']==va[0]['teacher_score']==.8


def test_temperature_is_loaded_from_matching_v2_probe(tmp_path,monkeypatch):
    features=tmp_path/'features';features.mkdir()
    (features/'run_config.json').write_text(json.dumps({'video_timing_policy':'sampled_fps_v2'}))
    report=tmp_path/'report.json';report.write_text(json.dumps({'features':str(features),'calibration':{'after':{'temperature':1.234}}}))
    args=['trainer','--mode','video_lora','--output',str(tmp_path/'out'),'--teacher-targets',str(tmp_path/'predictions.jsonl'),
        '--teacher-probe-report',str(report)]
    monkeypatch.setattr(sys,'argv',args)
    parsed=trainer.parse_args()
    assert parsed.video_layers==12 and parsed.teacher_calibration_temperature==1.234
    assert parsed.teacher_subset=='tav'
    (features/'run_config.json').write_text('{}')
    with pytest.raises(SystemExit):trainer.parse_args()


def test_first_epoch_interruption_is_archived_before_restart(tmp_path):
    output=tmp_path/'students/video_lora_seed13';output.mkdir(parents=True)
    (output/'run_config.json').write_text('{}')
    command=queue.student_command(tmp_path,'video_lora',13,output)
    assert not output.exists() and '--resume' not in command
    assert len(list(output.parent.glob('video_lora_seed13_interrupted_before_epoch1_*')))==1


def test_completed_epoch_uses_resume(tmp_path):
    output=tmp_path/'students/video_lora_seed13';output.mkdir(parents=True)
    (output/'run_config.json').write_text('{}');(output/'last.pt').write_bytes(b'checkpoint')
    assert '--resume' in queue.student_command(tmp_path,'video_lora',13,output)
    assert output.exists()


def test_full_video_parameter_and_gradient_guard(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'status':'complete'}))
    report={'mode':'video_lora','official_test_evaluated':False,'video_trainable_parameters':589824}
    (tmp_path/'report.json').write_text(json.dumps(report))
    grads={f'layer.{n}.lora_B.weight':.1 for n in range(48)}
    (tmp_path/'video_gradient_audit.json').write_text(json.dumps(grads))
    queue.validate_student(tmp_path,'video_lora',smoke=True)
    grads['layer.0.lora_B.weight']=0
    (tmp_path/'video_gradient_audit.json').write_text(json.dumps(grads))
    with pytest.raises(AssertionError):queue.validate_student(tmp_path,'video_lora',smoke=True)


@pytest.mark.parametrize('resume_completed_seed', [None, 13])
@pytest.mark.parametrize('controls_enabled', [False, True])
def test_priority_students_finish_before_any_control(tmp_path,monkeypatch,resume_completed_seed,controls_enabled):
    (tmp_path/'logs').mkdir()
    completed=set()
    if resume_completed_seed is not None:
        name=f'video_lora_seed{resume_completed_seed}'
        output=tmp_path/'students'/name
        output.mkdir(parents=True)
        (output/'status.json').write_text(json.dumps({'status':'complete'}))
        completed.add(name)
    launched=[]
    running=set()
    priority={f'video_lora_seed{s}' for s in queue.SEEDS}

    class Child:
        def __init__(self,command,**kwargs):
            self.name=command[0]
            if not self.name.startswith('video_lora_'):
                assert priority<=completed
            assert len(running)<2
            running.add(self.name)
            launched.append(self.name)
            self.pid=len(launched)
            self.remaining=4 if self.name=='video_lora_seed2026' else 1

        def poll(self):
            self.remaining-=1
            if self.remaining>0:return None
            running.discard(self.name)
            completed.add(self.name)
            return 0

    monkeypatch.setattr(queue.subprocess,'Popen',Child)
    monkeypatch.setattr(queue,'gpu_idle',lambda gpu:(True,{}))
    monkeypatch.setattr(queue,'verify_sources',lambda plan:None)
    monkeypatch.setattr(queue,'validate_student',lambda *args:None)
    monkeypatch.setattr(queue,'student_command',lambda base,mode,seed,output:[output.name])
    monkeypatch.setattr(queue,'status',lambda *args,**kwargs:None)
    monkeypatch.setattr(queue.time,'sleep',lambda seconds:None)
    queue.run_students(tmp_path,{'student_schedule':{'priority_modes':['video_lora'],
        'deferred_modes':['frozen_video','ta_only'],'complete_priority_before_controls':True,
        'controls_enabled':controls_enabled}})
    modes=queue.MODES if controls_enabled else ['video_lora']
    expected={f'{mode}_seed{seed}' for mode in modes for seed in queue.SEEDS}
    assert completed==expected and not running
    assert len(launched)==len(expected)-(resume_completed_seed is not None)


def test_primary_summary_does_not_require_deferred_controls(tmp_path,monkeypatch):
    (tmp_path/'docs').mkdir()
    monkeypatch.setattr(queue,'ROOT',tmp_path)
    def report(output,mode):
        assert mode=='video_lora'
        return {'mode':mode,'seed':int(output.name.rsplit('seed',1)[1]),'diagnostics':[],
            'valid_metrics':{'mae':.5,'pearson':.8,'acc2_nonzero':.9}}
    monkeypatch.setattr(queue,'validate_student',report)
    queue.summarize(tmp_path,modes=['video_lora'])
    result=json.loads((tmp_path/'summary.json').read_text())
    assert len(result['runs'])==3 and result['comparisons']=={}
    assert result['full_matrix_complete'] is False
    assert result['deferred_modes']==['frozen_video','ta_only']
    assert 'controls deferred by user' in (tmp_path/'docs/video_source_v2_results.md').read_text()
