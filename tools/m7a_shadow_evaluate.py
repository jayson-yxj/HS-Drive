"""Evaluator-only GT association and replay reproducibility; never imported by inference."""
import json
from pathlib import Path
from dataclasses import asdict,fields
import numpy as np
from datasets.m7a_contracts import *
from datasets.m7a_shadow import ReplayQueue
from tools.m7a_common import read,write,asset,provenance


def main():
    forward=read('reports/m7a_shadow_rgb_forward.json');audited=read('reports/m7a_shadow_rgb_forward_audited.json')
    cc=read('runs/m5o_cache_train/candidates.json');sup=read('runs/m5o_cache_train/supervision.json')
    labels={r['candidate_id']:r for r in sup['labels']};byid={r['candidate_id']:r for r in cc}
    records=[];equivalence=[];queue_checks=[]
    for run,other in zip(forward['records'],audited['records']):
        assert run['sample_id']==other['sample_id']
        assert [h['candidate_id'] for h in run['hypotheses']]==[h['candidate_id'] for h in other['hypotheses']]
        for h,hh in zip(run['hypotheses'],other['hypotheses']):
            error=float(np.max(abs(np.array(h['position_ego_xyz_m'])-hh['position_ego_xyz_m'])))
            equivalence.append(dict(candidate=h['candidate_id'],repeat_max_abs_m=error))
            old=byid[h['candidate_id']];assert list(h['region'])==old['region']
            ll=labels[h['candidate_id']]
            ref=ll.get('reference')
            records.append(dict(candidate=h['candidate_id'],sample_id=run['sample_id'],camera=h['camera'],
              original_frame=run['stamp']['frame_id'],timestamp=run['stamp']['timestamp_s'],matched=ll['matched'],
              association_status='historical_M5O_unique_reference' if ll.get('train_eligible') else 'UnmatchedOrAmbiguous_UnknownNotPhysicalFP',
              predicted_position=h['position_ego_xyz_m'],reference_ActorBoxCenter_evaluator_only=None if ref is None else ref['box_center_ego'],
              xy_error_m=None if ref is None else float(np.linalg.norm(np.array(h['position_ego_xyz_m'])[:2]-ref['box_center_ego'][:2])),
              quality='CandidateNoGo',safety_free=False))
        p=read(next(r['path'] for r in forward['models']['packets'] if read(r['path'])['sample_id']==run['sample_id']))
        q=ReplayQueue(2);q.push(p,p['timestamp_s'])
        hypothetical_now=p['timestamp_s']+run['inference_ms']/1000
        delivered=q.pop(hypothetical_now)
        queue_checks.append(dict(sample_id=run['sample_id'],compute_age_s=run['inference_ms']/1000,
          original_valid_until=p['valid_until_s'],result='ExpiredDropped' if delivered is None else 'WithinOfflineComputeAge',
          dropped=q.dropped,kind='OfflineTimingSimulation_not_live_control'))
    errors=[r['xy_error_m'] for r in records if r['xy_error_m'] is not None]
    a=Stamp(1,1,1.2,EGO);visual=RGBSpatialEvidence(a,'x',(),'sha');unknown=Unknown('unseen')
    try:PlanningRequest(visual,((0.,0.,0.),(1.,0.,0.)),'x',(1.,0.,0.)).validate(1);isolated=False
    except TypeError:isolated=True
    write('reports/m7a_oracle_vision_isolation.json',dict(kind='ActualFrozenGPUReplay_OfflineGTOnlyEvaluation_SyntheticTypeChecks',
      provenance=provenance(),status='InterfaceVerified_VisualQualityCandidateNoGo',
      inference_data_read_audit=asset('reports/m7a_execution_logs/inference_file_access.json'),
      separate_inference_process='tools.m7a_shadow_infer; no GT path in manifest or function signature',
      source_domains=dict(OracleSceneState='Privileged',PlanningRequest='Privileged_OfflinePlanning',
        RGBSpatialEvidence='RGB_estimate_OfflineShadow',DetectedObjectHypothesis='CandidateNoGo',Unknown='never_Free'),
      visual_request_rejected=isolated,control_adapter='AlwaysRaises_NoConversionImplemented',
      hypotheses_fields=[f.name for f in fields(DetectedObjectHypothesis)],no_oracle_object_reference=True,
      numerical_repeat=equivalence,max_repeat_error_m=max(e['repeat_max_abs_m'] for e in equivalence),
      all_candidates=records,matched=len(errors),unmatched_unknown=len(records)-len(errors),
      matched_xy_errors_m=errors,queue_age_simulation=queue_checks,
      qualification='No accuracy gate; two historical Train samples, no generalization',
      dense_BEV='CandidateNoGo_historical_model_unchanged_NotRun_this_replay',
      future_visual_control='NotApproved',GT_reference='ActorOBBCenter_not_mesh_or_contact'))
    print('Repeated',len(records),'candidates; matched',len(errors),'unmatchedUnknown',len(records)-len(errors))

if __name__=='__main__':main()
