"""Scene/actor/distance results with raw support and no testmodelselection."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash
from diagnostics.m5f_metrics import metrics


def actor_summary(actors):
    eligible=[a for a in actors if a['pixels']>=3];pixels=sum(a['pixels'] for a in actors);hits=sum(a['predicted_pixels'] for a in actors)
    return dict(observations=len(actors),eligible=len(eligible),zero_support=sum(a['pixels']==0 for a in actors),detected=sum(a['detected'] for a in eligible),detection_rate=sum(a['detected'] for a in eligible)/len(eligible) if eligible else None,pixels=pixels,pixel_recall=hits/pixels if pixels else None)


def main():
    dest=Path('reports/m5f_results_digest.json')
    if dest.exists():raise FileExistsError(dest)
    pix=json.loads(Path('reports/m5f_2d_presence_absence.json').read_text());geometry=json.loads(Path('reports/m5f_2d_3d_factorization.json').read_text());out=[]
    for r in pix['records']:
        actors=[a for f in r['frames'] for a in f['actors']];pairs=r['pairs'];n=sum(p['n'] for p in pairs);nr=sum(p['removed_n'] for p in pairs);unique=[]
        for frame in r['frames']:
            for aid in sorted({a['actor_id'] for a in frame['actors']}):
                aa=[a for a in frame['actors'] if a['actor_id']==aid];unique.append(dict(eligible=any(a['pixels']>=3 for a in aa),detected=any(a['detected'] for a in aa),visible_pixels=sum(a['pixels'] for a in aa)))
        strata={}
        for lo,hi in [(0,5),(5,10),(10,15),(15,20),(20,35),(35,80)]:strata[f'distance_{lo}_{hi}']=actor_summary([a for a in actors if lo<=a['distance_m']<hi])
        for typ in sorted({a['type_id'] for a in actors}):strata[typ]=actor_summary([a for a in actors if a['type_id']==typ])
        for size in ['small_le32','medium_le128','large']:strata[size]=actor_summary([a for a in actors if a['size_bucket']==size])
        paired=dict(n=n,presence_recall=sum(p['presence_hits'] for p in pairs)/n if n else None,empty_region_fpr=sum(p['empty_fp'] for p in pairs)/n if n else None,both_correct=sum(p['both_correct'] for p in pairs)/n if n else None,mean_rise=sum((p['mean_rise'] or 0)*p['n'] for p in pairs)/n if n else None,direction_rise=sum((p['direction_rise'] or 0)*p['n'] for p in pairs)/n if n else None,removed_n=nr,removed_fpr=sum(p['removed_fp'] for p in pairs)/nr if nr else None,mean_removed=sum((p['mean_removed'] or 0)*p['removed_n'] for p in pairs)/nr if nr else None,direction_fall=sum((p['direction_fall'] or 0)*p['removed_n'] for p in pairs)/nr if nr else None)
        out.append(dict(name=r['name'],seed=r['seed'],threshold=r['threshold'],pixels=r['strata']['pooled/all'],empty=r['strata'].get('empty'),actor_observations=actor_summary(actors),actor_unique_configurations=dict(total=len(unique),eligible=sum(a['eligible'] for a in unique),detected=sum(a['detected'] for a in unique),zero_support=sum(a['visible_pixels']==0 for a in unique)),actor_strata=strata,paired=paired,by_scene={k:v for k,v in r['strata'].items() if k.startswith('Town') and k.endswith('/all')}))
    geo=[]
    for name,seed,mode in sorted({(r['name'],r['seed'],r['mode']) for r in geometry['records']}):
        rr=[r for r in geometry['records'] if (r['name'],r['seed'],r['mode'])==(name,seed,mode)];actors=[a for r in rr for a in r['actors']];n=sum(a['visible_cells'] for a in actors);hist=sum(np.array(r['histogram']) for r in rr);th=rr[0]['threshold'];match={}
        for tol in ['0.5','1.0']:
            count=sum(a['matching']['tolerances'][tol]['matched_gt'] for a in actors);match[tol]=count/n if n else None
        bands={}
        for lo,hi in [(5,10),(10,15),(15,20),(20,35)]:
            aa=[a for a in actors if lo<=a['distance_m']<hi];nn=sum(a['visible_cells'] for a in aa);bands[f'{lo}_{hi}']=dict(n=nn,exact_recall=sum(a['exact_hits'] for a in aa)/nn if nn else None,recall_1m=sum(a['matching']['tolerances']['1.0']['matched_gt'] for a in aa)/nn if nn else None)
        geo.append(dict(name=name,seed=seed,mode=mode,threshold=th,pixelsurface_metrics=metrics(hist,th),actor_cells=n,actor_exact_recall=sum(a['exact_hits'] for a in actors)/n if n else None,actor_tolerant_recall=match,unmatched_actor_count=sum(a['visible_cells']>=3 and a['matching']['tolerances']['1.0']['matched_gt']==0 for a in actors),eligible_actors=sum(a['visible_cells']>=3 for a in actors),by_distance=bands,mean_unknown_high=float(np.mean([r['unknown_high_fraction'] for r in rr])),feature_rays=sum(r['actor_feature_rays'] for r in rr)))
    atomic_json(dest,dict(kind='offline_aggregate_actual_GPU_test_reports',pixels=out,geometry=geo,source_sha256=file_hash(__file__),limits='Microcounts correlated; independentunit3positiongroups. Unmatchedactor weakvisible-surface metric, not tracking. HistAP approximate. GeometricOracle sharedrenderer. No thresholdtuning.'))

if __name__=='__main__':main()
