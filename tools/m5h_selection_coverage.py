"""Selected checkpoints may precede full exposure even when full budget covers all."""
import json
from pathlib import Path
from collections import Counter
from simulator.recorder import atomic_json,file_hash

def main():
    dest=Path('reports/m5h_selected_exposure.json')
    if dest.exists():raise FileExistsError(dest)
    cov=json.loads(Path('reports/m5h_training_coverage.json').read_text());models=json.loads(Path('reports/m5h_distillation_ablation.json').read_text())['records'];records=[]
    for r in models:
        c=next(x for x in cov['records']if x['seed']==r['seed']);n=r['selected']['step'];counts=Counter(tuple(x)for x in c['schedule'][:n]);records.append(dict(name=r['name'],seed=r['seed'],selected_step=n,unique_views_at_selection=len(counts),unseen_at_selection=c['available_views']-len(counts),full_budget_unique=c['unique_views'],exposure=[dict(v,updates_at_selection=counts[(v['index'],v['camera_index'])])for v in c['exposure']]))
    atomic_json(dest,dict(kind='offline_actual_selected_checkpoint_exposure',records=records,source_sha256=file_hash(__file__),limits='Full2400coverage doesnotimply selected1200checkpoint sawall1248views. Val selectionunchanged; cannotclaim completeexposure excludesallcoverage limitations.'))
if __name__=='__main__':main()
