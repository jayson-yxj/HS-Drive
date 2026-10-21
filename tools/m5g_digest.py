"""Human-scale aggregate with independent position units and all failures retained."""
import json
from pathlib import Path
import numpy as np
from diagnostics.m5f_metrics import metrics
from simulator.recorder import atomic_json,file_hash


def paired(ps):
    n=sum(x['n']for x in ps);nr=sum(x['removed_n']for x in ps)
    return dict(n=n,removed_n=nr,presence_recall=sum(x['present_hits']for x in ps)/max(n,1),removed_fpr=sum(x['removed_fp']for x in ps)/max(nr,1),both_correct=sum(x['both_correct']for x in ps)/max(n,1),mean_rise=sum((x['mean_rise']or 0)*x['n']for x in ps)/max(n,1),direction_fall=sum((x['direction_fall']or 0)*x['removed_n']for x in ps)/max(nr,1))


def main():
    out=Path('reports/m5g_results_digest.json')
    if out.exists():raise FileExistsError(out)
    p=json.loads(Path('reports/m5g_pixel_test.json').read_text());pixels=[]
    for r in p['records']:
        a=r['result'];actors=a['actors'];scenes={g:paired([x for x in a['pairs']if x['group']==g])for g in a['by_scene']};bands={}
        for lo,hi in [(0,5),(5,10),(10,15),(15,20),(20,35),(35,80)]:
            aa=[x for x in actors if lo<=x['distance_m']<hi and x['pixels']>=3];bands[f'{lo}_{hi}']=dict(eligible=len(aa),detected=sum(x['detected']for x in aa))
        pixels.append(dict(name=r['name'],seed=r['seed'],threshold=a['threshold'],metrics=a['metrics'],supported=r['supported']['metrics'],vehicle=r['vehicle']['metrics'],pedestrian=r['pedestrian']['metrics'],paired=paired(a['pairs']),by_scene=a['by_scene'],empty_by_scene=a['empty_by_scene'],pair_by_scene=scenes,actor_bands=bands,actor_eligible=sum(x['pixels']>=3 for x in actors),actor_detected=sum(x['detected']for x in actors),small_eligible=sum(3<=x['pixels']<=32 for x in actors),small_detected=sum(3<=x['pixels']<=32 and x['detected']for x in actors)))
    pre=json.loads(Path('reports/m5g_pretrained_test.json').read_text());pretrained={k:dict(metrics=r['metrics'],threshold=r['threshold'],paired=paired(r['pairs']),by_scene=r['by_scene'],eligible=sum(x['pixels']>=3 for x in r['actors']),detected=sum(x['detected']for x in r['actors']))for k,r in pre['results'].items()};shadow=json.loads(Path('reports/m5g_shadow_comparison.json').read_text());bev=[]
    for seed,name in sorted({(r['seed'],r['mode'])for r in shadow['records']}):
        rr=[r for r in shadow['records']if(r['seed'],r['mode'])==(seed,name)];n=sum(r['matching']['tolerances']['1.0']['gt_count']for r in rr);hit=sum(r['matching']['tolerances']['1.0']['matched_gt']for r in rr);h=sum(np.array(r['histogram'])for r in rr);bev.append(dict(seed=seed,mode=name,metrics=metrics(h,.2),gt_cells=n,recall_1m=hit/max(n,1),unknown_high_fraction=float(np.mean([r['unknown_high_fraction']for r in rr])),by_scene={g:dict(n=sum(r['matching']['tolerances']['1.0']['gt_count']for r in rr if r['group']==g),hits=sum(r['matching']['tolerances']['1.0']['matched_gt']for r in rr if r['group']==g))for g in sorted({r['group']for r in rr})}))
    atomic_json(out,dict(kind='offline_aggregate_actual_frozen_test_results',pixels=pixels,pretrained=pretrained,bev=bev,source_sha256=file_hash(__file__),limits='Threepositiongroups; pixels/actorconfigurations correlated. Pretrainedunsupportedcone excluded. Newdepthdiagnostic failednearqualitygate notdeployable. No results imply safetyFree.'))

if __name__=='__main__':main()
