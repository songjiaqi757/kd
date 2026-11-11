"""Completion gate for Phase 0; fails rather than declaring partial work complete."""
import csv
from common import *

def main():
    plan=read(OUT/'protocol/plan.json');summary=read(OUT/'status/summary.json')
    assert summary['status']=='complete',summary
    assert summary['prediction_exports']==72 and summary['gradient_cases']==12
    assert read(OUT/'status/probe_replay.json')['passed']
    assert read(OUT/'status/feature_fingerprints.json')['all_four_feature_caches_match']
    assert read(OUT/'status/checkpoint_inventory.json')['all_current_files_hashed']
    expected_counts={'teacher_subset_quality.csv':112,'probe_agreement.csv':112,'probe_task_quality.csv':168,'student_absorbability.csv':504,'interaction_coordinate_error.csv':504,'interaction_order_error.csv':216,'student_task_quality.csv':72,'gradient_compatibility.csv':2772,'sample_quadrants.csv':2592,'validation_replay.csv':36,'coverage.csv':72,'probe_checkpoint_replay.csv':12}
    counts={}
    for name,count in expected_counts.items():
        data=list(csv.DictReader((OUT/'tables'/name).open()));counts[name]=len(data);assert len(data)==count,(name,len(data),count)
        if name=='validation_replay.csv':
            for r in data:assert float(r['mae_abs_delta'])<=float(r['tolerance']) and r['parameter_updates']=='0' and r['parameters_unchanged']=='True'
    checked=[]
    for c in plan['cases']:
        path=Path(c['checkpoint']);assert sha(path)==c['checkpoint_sha256'],f'Original checkpoint changed: {path}'
        for split in ['train','valid']:
            dest=OUT/f'predictions/{c["id"]}_{split}.jsonl';report=read(dest.with_suffix('.report.json'))
            assert report['sha256']==sha(dest);assert report['parameter_updates']==0;assert report['parameter_versions_unchanged']
        checked.append(c['id'])
    for path in (OUT/'gradients').glob('*.json'):
        result=read(path);assert len(result['samples'])==6 and result['parameter_updates']==0 and result['parameter_versions_unchanged']
    # Freeze the actual audit implementation and deliverable hashes after successful checks.
    scripts={p.name:sha(p) for p in (OUT/'source').glob('*.py')}
    artifacts={str(p.relative_to(OUT)):sha(p) for folder in ['tables','figures','protocol','decisions'] for p in (OUT/folder).rglob('*') if p.is_file() and p.name!='queue.lock'}
    result=dict(status='complete',phase=0,optimizer_steps=0,official_test_predictions_used=False,source_sha256=scripts,artifact_sha256=artifacts,verified_cases=checked,table_counts=counts,original_checkpoint_sha256_unchanged=True,teacher_probe_replay_passed=True)
    save(OUT/'status/final_verification.json',result)
    print(json.dumps(dict(status='complete',phase=0,verified_cases=len(checked),table_counts=counts)),flush=True)

if __name__=='__main__':main()
