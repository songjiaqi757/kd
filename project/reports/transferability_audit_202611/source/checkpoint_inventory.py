"""0A: actual hashes and validation metrics of all retained files in scoped runs."""
import gc
import torch
from common import *

def main():
    torch.set_num_threads(1);plan=read(OUT/'protocol/plan.json');records=[];coverage=[]
    cache={c['checkpoint']:c['checkpoint_sha256'] for c in plan['cases']}
    for c in plan['cases']:
        run=Path(c['run']);history=read(run/'history.json')
        if isinstance(history,dict):history=history.get('epochs',history.get('history'))
        hist={int(r['epoch']):r for r in history};files={}
        for pattern in ['*.pt','checkpoints/*.pt']:
            for p in run.glob(pattern):
                if p.is_file():files.setdefault(p.resolve(),[]).append(str(p))
        inventory_path=run/'checkpoint_inventory.json';old={}
        if inventory_path.exists():old={int(r['epoch']):r for r in read(inventory_path).get('epochs',[])}
        present=set()
        for p,aliases in sorted(files.items()):
            before=p.stat();state=torch.load(p,map_location='cpu',mmap=True,weights_only=False);epoch=int(state['epoch']);del state;gc.collect()
            digest=next((cache[x] for x in aliases if x in cache),None) or sha(p)
            after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            present.add(epoch);r=hist.get(epoch,{})
            records.append(dict(case=c['id'],dataset=c['dataset'],method=c['method'],seed=c['seed'],epoch=epoch,path=str(p),aliases=json.dumps(aliases),bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=digest,selected_validation_mae=epoch==c['epoch'],historical_archive_sha256=old.get(epoch,{}).get('sha256'),hash_scope='current file computed; historical archive hash may differ after optimizer stripping/reserialization',**r.get('valid_metrics',{})))
        for epoch,r in hist.items():coverage.append(dict(case=c['id'],dataset=c['dataset'],method=c['method'],seed=c['seed'],epoch=epoch,current_weight_retained=epoch in present,selected_validation_mae=epoch==c['epoch'],historical_archive_sha256=old.get(epoch,{}).get('sha256'),**r['valid_metrics']))
        print(json.dumps(dict(case=c['id'],current_files=len(files),epochs_with_weights=len(present),history_epochs=len(hist))),flush=True)
    table('checkpoint_inventory_audit.csv',records)
    table('checkpoint_history_coverage.csv',coverage)
    save(OUT/'protocol/retained_checkpoint_manifest.json',records)
    save(OUT/'status/checkpoint_inventory.json',dict(status='complete',current_files=len(records),history_epochs=len(coverage),all_current_files_hashed=True))

if __name__=='__main__':main()
