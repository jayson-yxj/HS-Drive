"""Aggregate frozen factorization with matching checkpoint/threshold, no tuning."""
import json
from pathlib import Path
import numpy as np
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5e_stride8_review.json')
    if dest.exists():raise FileExistsError(dest)
    source=json.loads(Path('reports/m5e_stride8_factorization.json').read_text());records=[]
    for r in source['records']:
        f=r['frames'];g=r['geometry'];w=list(r['witness']['by_scene'].values());n=sum(x['n'] for x in w);rays=sum(x['feature_rays'] for x in g);times={}
        for key in ['full_forward_ms','splat_ms','bev_and_head_ms']:
            t=[v for x in r['resources'] for v in x[key]];times[key]=dict(mean=float(np.mean(t)),p95=float(np.quantile(t,.95)),max=float(np.max(t)),n=len(t))
        records.append(dict(trained=r['trained'],seed=r['seed'],mode=r['mode'],threshold=r['threshold'],feature_rays=rays,actor_exact_recall=sum(x['actor_hits'] for x in f)/sum(x['actor_n'] for x in f),ground_fpr=sum(x['ground_fp'] for x in f)/sum(x['ground_n'] for x in f),unknown_high_fraction=sum(x['unknown_high'] for x in f)/sum(x['unknown_n'] for x in f),gold_presence=sum(x['presence_hits'] for x in w)/n,gold_absence_fpr=sum(x['absence_false_positives'] for x in w)/n,gt_1m_mass=sum(x['gt_1m_mass_sum'] for x in g)/rays,times=times,peak_mib=max(x['peak_allocated_mib'] for x in r['resources']),limits='Allocation includes resident dataset/cache; fullforward p8 modes compute bothencoderpaths. BEV+head timing bundled, separate BEV-only supplementary timing report.'))
    atomic_json(dest,dict(kind='offline_aggregation_actual_RTX4090_factorized_validation',records=records,source_sha256=file_hash(__file__),limits='Factorized diagnostic not strict singlevariable highresolution training. No threshold changes. DifferentUV masks betweenS16/P8; feature/downsample path controls raycount but featuredistribution shifts.'))

if __name__=='__main__':main()
