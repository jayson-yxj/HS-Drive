"""Frozen A/B evaluator, preserving natural denominators and image identity gates."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from diagnostics.m8b_metrics import natural,conditional,conditional_matrix,aggregate_natural,errors,coarse
from diagnostics.m8d_evaluation import load_masks,grade_frame,paired_counts,localization
from tools.m8b_evaluate import loc_summary
from tools.m8d_common import *

def main():
    ledger=json.loads((BASELINE/'evaluation_ledger.json').read_text())['frames']
    index={x['token']:x for x in json.loads((BASELINE/'evaluation_index.json').read_text())}
    caches={s:json.loads(Path('runs/m5o_cache_'+s+'/supervision.json').read_text())
            for s in ['train','val','calibration','sealed']}
    original_identity={(r['token'],r['gt_index']):r for r in json.loads(
        Path('reports/m8c_2d_3d_association_audit.json').read_text())['rows']}
    frames=[];targets=[];identities={'A':[],'B':[]};proxy={'A':[],'B':[]}
    for f in ledger:
        token=f['token'];packet=json.loads(Path(index[token]['packet']['path']).read_text())
        predictions={'A':json.loads((BASELINE/'forward'/(token+'.json')).read_text()),
                     'B':json.loads((ROOT/'full'/(token+'_B.json')).read_text())}
        support=load_masks(f,index[token],caches[f['split']])
        entry=dict(token=token,group=f['group'],split=f['split'],map=f['map'],
                   frame_id=packet['frame_id'],timestamp_s=packet['timestamp_s'],variants={})
        for variant,p in predictions.items():
            n=natural(p,f['gts'],.1,2.);n4=natural(p,f['gts'],.1,4.)
            if variant=='A':
                assert n==f['curves']['0.1']['natural']
                pairs=[[r['prediction_index'],r['gt_index']] for r in f['curves']['0.1']['conditional']]
            else:
                pairs=conditional(p,f['gts'],conditional_matrix(p,f['gts'],packet),.1)
            grades=grade_frame(p,f,packet,pairs,support)
            if variant=='A':
                for r in grades:
                    old=original_identity[(token,r['gt_index'])]
                    assert r['grade']==old['grade'] and r['prediction_index']==old['prediction_index']
                    if r['metric_error'] is not None:
                        assert r['metric_error']==old['metric_error']
            identities[variant].extend(grades)
            proxy[variant].extend(dict(token=token,group=f['group'],split=f['split'],**r)
                                  for r in errors(p,f['gts'],pairs,packet))
            entry['variants'][variant]=dict(natural=n,sensitivity4m=n4,projection_pairs=pairs,
                scores=p['scores'],labels=p['labels'],output=asset((BASELINE/'forward'/(token+'.json')) if variant=='A' else ROOT/'full'/(token+'_B.json')))
        for gi,g in enumerate(f['gts']):
            distance=float(np.linalg.norm(g['box'][:2]))
            targets.append(dict(token=token,gt_index=gi,actor_id=g['actor_id'],group=f['group'],
                split=f['split'],map=f['map'],type=g['type'],range_m=distance,
                range_bin='0-10' if distance<10 else '10-20' if distance<20 else '20-40' if distance<40 else '40+',
                cameras=[v['camera'] for v in g['views']],
                A=any(gi==j for i,j in entry['variants']['A']['natural']['pairs']),
                B=any(gi==j for i,j in entry['variants']['B']['natural']['pairs'])))
        frames.append(entry)
        print(token,'evaluated',flush=True)
    assert len(frames)==192 and len(targets)==155 and len({f['group'] for f in frames})==38
    meta=provenance('OfflinePairedEvaluation_RecordedCARLAGT_NotNewClosedLoop')
    strata=defaultdict(list)
    for r in targets:
        keys=['all','class/'+r['type'],'range/'+r['range_bin'],'group/'+r['group'],
              'split/'+r['split'],'map/'+r['map']]
        keys+=['camera/'+c for c in r['cameras']]
        keys+=['direction/'+d for d in set(('front' if c.startswith('front') else 'rear') for c in r['cameras'])]
        keys+=['side/'+s for s in set('left' if 'left' in c else 'right' if 'right' in c else 'center' for c in r['cameras'])]
        for k in keys:strata[k].append(r)
    quality={}
    for key,rr in strata.items():
        quality[key]=dict(**paired_counts(rr),groups=len({r['group'] for r in rr}),
                         statistical_scope='Descriptive location-correlated historical samples; no independent final test')
    group_results=[]
    for group in sorted({f['group'] for f in frames}):
        ff=[f for f in frames if f['group']==group];rr=[r for r in targets if r['group']==group]
        group_results.append(dict(group=group,split=ff[0]['split'],frames=len(ff),**paired_counts(rr),
            A=aggregate_natural([f['variants']['A']['natural'] for f in ff]),
            B=aggregate_natural([f['variants']['B']['natural'] for f in ff])))
    write('reports/m8d_paired_detection_quality.json',dict(**meta,score=.1,xy_gate_m=2,z_gate_m=2,
        natural={v:aggregate_natural([f['variants'][v]['natural'] for f in frames]) for v in ['A','B']},
        sensitivity4m={v:aggregate_natural([f['variants'][v]['sensitivity4m'] for f in frames]) for v in ['A','B']},
        strata=quality,all38_groups=group_results,targets=targets,
        unmatched_definition='FP key inherited from evaluator means Annotation-Unmatched, NOT confirmed physical FP'))
    byid={v:{(r['token'],r['gt_index']):r for r in identities[v]} for v in ['A','B']}
    common={k for k,r in byid['A'].items() if r['grade']=='IdentitySupported' and byid['B'][k]['grade']=='IdentitySupported'}
    fixed={k for k,r in byid['A'].items() if r['grade']=='IdentitySupported'}
    identity_groups={}
    selectors={'all':lambda r:True,'far20to40':lambda r:20<=r['range_m']<40}
    selectors.update({s:lambda r,s=s:r['split']==s for s in ['train','val','calibration','sealed']})
    selectors.update({c:lambda r,c=c:r['type']==c for c in ['vehicle','person','cone']})
    for key,select in selectors.items():
        identity_groups[key]={v:dict(Counter(r['grade'] for r in identities[v] if select(r))) for v in ['A','B']}
    write('reports/m8d_paired_identity_support.json',dict(**meta,rule=CFG['identity_rule'],groups=identity_groups,
        common_supported=[list(k) for k in sorted(common)],A_fixed_six=[dict(key=list(k),A=byid['A'][k],B=byid['B'][k]) for k in sorted(fixed)],
        all_rows=identities,limit='Evidence-supported automated image association, not certified human identity. No GT center error used for grade.'))
    loc={key:{v:localization([r for r in identities[v] if select(r)],stats) for v in ['A','B']}
         for key,select in selectors.items()}
    loc['common_identity']={v:localization([byid[v][k] for k in sorted(common)],stats) for v in ['A','B']}
    loc['fixed_A_six']={v:localization([byid[v][k] for k in sorted(fixed)],stats) for v in ['A','B']}
    write('reports/m8d_paired_localization.json',dict(**meta,strict=loc,
        historical_proxy={v:dict(n=len(proxy[v]),metrics=loc_summary(proxy[v])) for v in ['A','B']},
        proxy_rows=proxy,strict_denominator_notice='Compare strict common subset separately. Historical A92/Bproxy are untrusted projection associations and not the same denominator as strict identity.',
        far_status='FarFieldIdentityInsufficient' if min(loc['far20to40'][v]['n'] for v in ['A','B'])<20 else 'DescriptiveOnly'))
    burden={}
    for v in ['A','B']:
        scores=[s for f in frames for s in f['variants'][v]['scores']]
        burden[v]=dict(raw_topK=len(scores),scores=stats(scores),
            score_by_class={str(c):stats([s for f in frames for s,l in zip(f['variants'][v]['scores'],f['variants'][v]['labels']) if l==c]) for c in range(10)},
            natural=aggregate_natural([f['variants'][v]['natural'] for f in frames]),
            empty_annotation_frames=[dict(token=f['token'],group=f['group'],count=f['variants'][v]['natural']['predictions'])
                                     for f in frames if f['variants'][v]['natural']['gt']==0])
    write('reports/m8d_score_candidate_burden.json',dict(**meta,variants=burden,
        annotation_status='M8-C incomplete background annotations persist; unmatched is Unknown, not physical FP or Free'))
    write(ROOT/'evaluation.json',dict(**meta,frames=frames,targets=targets,identities=identities))
    print(json.dumps({k:quality[k] for k in ['all','range/20-40']}))
if __name__=='__main__':main()
