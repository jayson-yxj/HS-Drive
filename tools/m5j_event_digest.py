"""Fixed-workpoint Empty/Present/Removed, matched Oracle and scene summaries."""
import json
from pathlib import Path
from tools.m5j_common import write
from tools.m5j_metrics import aggregate
from simulator.recorder import file_hash

def main():
    raw=json.loads(Path('reports/m5j_aggregation_test.json').read_text());detail=json.loads(Path('reports/m5j_spatial_details_test.json').read_text());sel=json.loads(Path('reports/m5j_selection.json').read_text());events=[];oracle=[];stability=[];shapes={}
    for r in raw['records']:shapes.setdefault(r['group'],set()).add(r['configuration'])
    for seed in [101,102]:
        for method in ['raw','gate','regularized']:
            t=sel['workpoints'][f'{seed}_{method}']['selected'];rr=[r for r in raw['records']if r['seed']==seed and r['method']==method and r['threshold']==t]
            for event in ['empty','present','removed']:
                sub=[r for r in rr if (r['configuration']==min(shapes[r['group']])if event=='empty'else r['configuration']==max(shapes[r['group']])if event=='removed'else r['configuration']not in [min(shapes[r['group']]),max(shapes[r['group']])])];events.append(dict(seed=seed,method=method,event=event,threshold=t,**aggregate(sub)))
            for mode in ['mobile_learned','oracle_learned','mobile_oracle','oracle_oracle']:
                sub=[r for r in detail['oracle_diagnostics']if r['seed']==seed and r['method']==method and r['mode']==mode];oracle.append(dict(seed=seed,method=method,mode=mode,threshold=t,**aggregate(sub)))
            for group in shapes:
                candidate=aggregate([r for r in rr if r['group']==group]);baseline=aggregate([r for r in raw['records']if r['seed']==seed and r['method']=='raw'and r['threshold']==.2 and r['group']==group]);stability.append(dict(seed=seed,method=method,group=group,precision_delta=candidate['precision']-baseline['precision'],recall1_delta=candidate['recall1']-baseline['recall1'],actor_detection_delta=candidate['actor_detection']-baseline['actor_detection'],passed_gain=candidate['precision']>=baseline['precision']+.05 and candidate['recall1']>=baseline['recall1']-.02))
    write('reports/m5j_event_oracle_digest.json',dict(kind='postfreeze_descriptive_locked_workpoint_aggregation',events=events,oracle=oracle,scenes=stability,source_sha256=file_hash(__file__),limits='Empty/Removed may include existing mapsemanticobjects; counts onlyconfirmedbackground. Strict physical absence notasserted. Oracle learnedfourwayGTmask matched separately from formalunmasked output. Testscenes notused toalterselection.'))
if __name__=='__main__':main()
