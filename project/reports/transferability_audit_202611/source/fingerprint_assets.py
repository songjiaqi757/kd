"""Verify teacher feature bytes and identify base encoder/teacher revisions."""
import sys
from common import *

def main():
    plan=read(OUT/'protocol/plan.json');features={}
    for d,p in plan['protocols'].items():
        a=read(p['assets']);original=read(p['original_assets']);path=Path(a['teacher_features'])
        expected=next(v for k,v in original['input_sha256'].items() if mapped(k)==str(path))
        actual=sha(path);assert actual==expected,(d,str(path),expected,actual)
        features[d]=dict(path=str(path),sha256=actual,bytes=path.stat().st_size,status='matches_frozen_teacher_feature_cache')
        print(json.dumps(dict(dataset=d,teacher_feature_sha256=actual,passed=True)),flush=True)
    base={}
    for name in ['Qwen3-0.6B-Base','WavLM-Base-Plus','VideoMAE-Base','Qwen3-Omni-30B-A3B-Instruct']:
        root=ROOT/'model'/name
        config={str(p.relative_to(root)):sha(p) for pattern in ['*.json','*.txt','*.model'] for p in root.glob(pattern) if p.is_file()}
        weights={str(p.relative_to(root)):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for pattern in ['*.safetensors','*.bin'] for p in root.glob(pattern) if p.is_file()}
        if '30B' not in name:
            for file,value in weights.items():value['sha256']=sha(root/file)
        base[name]=dict(path=str(root),config_sha256=config,weights=weights,
            scope='30B shard metadata + config; actual frozen hidden arrays verified byte-for-byte' if '30B' in name else 'base encoder weight SHA256 + config')
    save(OUT/'protocol/feature_and_base_model_fingerprints.json',dict(features=features,models=base))
    save(OUT/'status/feature_fingerprints.json',dict(status='complete',all_four_feature_caches_match=True))

if __name__=='__main__':main()
