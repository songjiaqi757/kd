import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/ai/sjq/kd')
OLD = ROOT / 'project/reports/transferability_audit_202611'
OUT = ROOT / 'project/reports/icd_review_revision_202611'
NAMES = ['t','a','v','ta','tv','av','tav']

def read(p):
    return json.loads(Path(p).read_text())

def rows(p):
    with Path(p).open() as f:
        return [json.loads(x) for x in f if x.strip()]

def save(p, value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,p)

def jsonl(p, values):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:
        for value in values:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def mapped(v):
    if isinstance(v,dict):return {mapped(k):mapped(x) for k,x in v.items()}
    if isinstance(v,list):return [mapped(x) for x in v]
    if isinstance(v,str):
        for prefix in ['/data/sjq/kd','/home/wy/sjq/kd']:
            if v==prefix or v.startswith(prefix+'/'):return str(ROOT)+v[len(prefix):]
    return v

def table(name, records, fields=None):
    p=OUT/'phase0_5'/name;p.parent.mkdir(parents=True,exist_ok=True)
    keys=fields or list(dict.fromkeys(k for row in records for k in row))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
