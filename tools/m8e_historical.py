"""Re-audit fixed historical candidates; preserve M8-C/D metrics and unresolved background."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from datasets.m8e_reference import prediction_grade
from diagnostics.m8d_evaluation import metric_error,localization
from tools.m8d_common import stats
from tools.m8e_common import *

def main():
    baseline=Path('/root/gpufree-data/hs-drive-experiments/m8b-petr-zero-shot')
    old=json.loads(Path('reports/m8c_2d_3d_association_audit.json').read_text())['rows']
    frames={f['token']:f for f in json.loads((baseline/'evaluation_ledger.json').read_text())['frames']}
    rows=[]
    for r in old:
        packet=json.loads((baseline/'rgb_bundle'/r['token']/'packet.json').read_text())
        raw=json.loads(Path(r['source_GT']['path']).read_text())
        actor=next(a for a in raw['actors'] if a['actor_id']==r['actor_id'])
        time_ok=actor.get('actor_snapshot_frame_id')==packet['frame_id']
        timestamp_ok=abs(actor.get('actor_snapshot_timestamp_s',-1)-packet['timestamp_s'])<1e-5
        views=[dict(v,source_linked=time_ok and timestamp_ok and v['GT_association_purity']>=.9) for v in r['views']]
        grade=prediction_grade(views,r['prediction_index'] is not None)
        error=None
        if grade.startswith('RenderLinked'):
            p=json.loads((baseline/'forward'/(r['token']+'.json')).read_text())
            error=metric_error(p['boxes'][r['prediction_index']],frames[r['token']]['gts'][r['gt_index']],packet,views)
        rows.append(dict(**{k:r[k] for k in ['token','frame_id','timestamp_s','split','group','actor_id','gt_index','type','range_m','prediction_index','score','source_GT']},
            historical_grade=r['grade'],grade=grade,views=views,metric_error=error,
            actor_snapshot_frame_ok=time_ok,actor_timestamp_ok=timestamp_ok,
            source_limit='Historical unique renderer/Actor OBB proxy, no complete EnvironmentObject snapshot. This is new evaluator evidence grading, not model improvement.'))
    old_bg=json.loads(Path('reports/m8c_annotation_completeness.json').read_text())['records']
    by_pred={(r['token'],r['prediction_index']):r for r in rows if r['prediction_index'] is not None}
    background=[]
    for r in old_bg:
        match=by_pred.get((r['token'],r['prediction_index']))
        grade=match['grade'] if match else 'UnlabeledOrUnknown'
        if r['status']=='DuplicateHypothesis':status='DuplicateHypothesis'
        elif grade.startswith('RenderLinked'):status='SimulatorEvidenceReviewed'
        elif r['coarse_class']=='unsupported_barrier':status='UnlabeledOrUnknown'
        else:status='AmbiguousIdentity' if grade=='AmbiguousIdentity' else 'UnlabeledOrUnknown'
        background.append(dict(original_output=dict(token=r['token'],prediction_index=r['prediction_index'],
                       score=r['score'],class_id=r['class_id'],output=asset(baseline/'forward'/(r['token']+'.json'))),
                historical_status=r['status'],status=status,render_identity_grade=grade,
                linked_recorded_actor_id=match['actor_id'] if match and grade.startswith('RenderLinked') else None,
                duplicate_query=r['duplicate_prediction'],views=r['views'],
                physical_false_positive_confirmed=False,
                unresolved='No historical complete environment inventory; later live world IDs cannot backfill an old episode. Semantic overlap cannot establish exact object identity.',
                review_type='SimulatorEvidenceReviewed',human_double_verified=False))
    write('reports/m8e_background_annotation_audit.json',dict(**provenance('OfflineFixedHistorical154CandidateAudit'),
        candidate_count=154,records=background,statuses=dict(Counter(r['status'] for r in background)),
        confirmed_physical_false_positives=0,
        barrier_policy='PETR supports barrier; old unsupported_barrier is missing annotation taxonomy, now Unknown unless asset identity is supported. No blanket FP.',
        static_backfill='NotAvailable for original frames; new live EnvironmentObject IDs not substituted into history'))
    groups={}
    for key,select in [('all',lambda r:True),('far20to40',lambda r:20<=r['range_m']<40),
        ('single',lambda r:r['grade']=='RenderLinkedSingleView'),('multi',lambda r:r['grade']=='RenderLinkedMultiView')]:
        rr=[r for r in rows if select(r)]
        groups[key]=dict(total=len(rr),grades=dict(Counter(r['grade'] for r in rr)),localization=localization(rr,stats))
    write('reports/m8e_historical_evaluator.json',dict(**provenance('OfflineHistoricalFrozenPredictions_NewIdentityGrading'),
        rows=rows,groups=groups,GT=155,unchanged_natural=45,old_strict_six_preserved=True,
        conclusion='Evaluator denominator change only, no PETR model improvement and no independent new final result.',
        unmatched_outputs='Entire original57600 remain accessible; fixed154background ledger is diagnostic sampling only.'))
    print('historical',groups['all']['grades'],'far',groups['far20to40']['grades'],'background',Counter(r['status'] for r in background))
if __name__=='__main__':main()
