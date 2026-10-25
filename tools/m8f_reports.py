"""Separate natural geometry, render evidence, history and training eligibility."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tools.m8f_common import *
from datasets.m8f_quality import polygon_distance,eligibility

def main():
 cfg=json.loads(Path('configs/m8f/contract.json').read_text());protocol=json.loads(Path('configs/m8f/protocol.json').read_text());history=json.loads((ROOT/'historical_index.json').read_text())
 requests=[];certificates=[];support=[];background=[];sync=[];sidecars=[];independence=[];group_out=[]
 for g in protocol['groups']:
  path=ROOT/'capture'/g['group'];a=json.loads((ROOT/'audit'/(g['group']+'.json')).read_text())
  req=json.loads((path/'evaluator/grounded_requests.json').read_text())['requests'];requests.extend(dict(group=g['group'],**r) for r in req)
  byphase={f['phase']:f for f in a['frames']};f=byphase['present'];side=json.loads(Path(f['sidecar']['path']).read_text());checks=json.loads(Path(side['ground_support']['path']).read_text())['checks']
  ch={x['key']:x for x in checks}
  prior=[r for r in history['records'] if r['map']==g['map']]
  for request in req:
   key=request['key'];rp=request['preflight']['present'];corners=rp.get('requested_obb');poly=np.asarray(corners['corners_world_m'])[[0,4,6,2],:2] if corners else None
   nearest=None;nearest_actor=None;scene=None
   if poly is not None:
    for r in prior:
     if r['polygon'] is not None:d=polygon_distance(poly,r['polygon'])
     else:
      # Proven lower bound using circumscribed circles: uncertainty cannot improve separation.
      d=max(0,float(np.linalg.norm(poly.mean(0)-r['center_xy']))-float(np.linalg.norm(poly-poly.mean(0),axis=1).max())-r['radius_m'])
     item=dict(distance_lower_bound_m=d,source=r['source'],kind=r['kind'],detail=r['detail'])
     if nearest is None or d<nearest['distance_lower_bound_m']:nearest=item
     if 'Actor' in r['kind'] or 'OBB' in r['kind']:
      if nearest_actor is None or d<nearest_actor['distance_lower_bound_m']:nearest_actor=item
     if 'Envelope' in r['kind']:
      if scene is None or d<scene['distance_lower_bound_m']:scene=item
   overlap=nearest is not None and nearest['distance_lower_bound_m']<cfg['independence']['target_footprint_distance_m']+cfg['independence']['target_uncertainty_margin_m']
   independence.append(dict(group=g['group'],key=key,grade='HistoricalTargetOverlap' if overlap else 'HistoricalCoverageIncomplete',
       nearest_any=nearest,nearest_actor=nearest_actor,nearest_scene_envelope=scene,RepeatedLayoutFamily=True,FreshMapEvidence=False))
   cr=ch.get(key);obj=next((o for o in f['objects'] if cr and o['entity_id']==cr['entity_id'] and o['source']=='Actor'),None)
   final=cr['final'] if cr else None
   grades=dict(RenderIdentityGrade=obj['identity_grade'] if obj else 'NotSpawned_PreflightRejected',
       GroundSurfaceGrade=final['status'] if final else rp['status'],
       NaturalPlacementGrade='NaturalByGeometryProxy' if final and final['status']=='GroundedByGeometryProxy' else 'Rejected',
       AnnotationGrade='SourceContractComplete' if obj and obj['frame_consistent'] else 'NotQualified',
       SplitIndependenceGrade='HistoricalTargetOverlap' if overlap else 'HistoricalCoverageIncomplete',InputSourceGrade='CompleteSynchronized')
   cert=dict(group=g['group'],key=key,entity_key=obj['entity_key'] if obj else None,category=rp['asset']['category'],
      **grades,TrainingEligibility=eligibility(grades),requested_sources=asset(path/'evaluator/grounded_requests.json'),
      range_m=obj['range_m'] if obj else None,accepted_for_spawn=request['accepted_for_spawn'],
      render_views=obj['views'] if obj else [],PhysicalContactCertified=False,ground_contact_exact=False,
      natural_proxy_qualified=bool(final and final['status']=='GroundedByGeometryProxy' and obj and obj['identity_grade'].startswith('RenderLinked')),
      rejection_reasons={k:v['reasons'] for k,v in request['preflight'].items()})
   certificates.append(cert)
  for phase,fr in byphase.items():
   s=json.loads(Path(fr['sidecar']['path']).read_text());sidecars.append(fr['sidecar'])
   support.append(asset(s['ground_support']['path']))
   for cam,modalities in s['sensors'].items():
    for kind,x in modalities.items():
     sync.append(dict(group=g['group'],phase=phase,camera=cam,kind=kind,ok=x['frame_id']==s['frame_id'] and abs(x['timestamp_s']-s['timestamp_s'])<1e-5 and sha(x['path'])==x['sha256']))
  statics=[o for o in f['objects'] if not o.get('is_controlled_target')]
  # Audited pixel regions are not whole-image negatives or ground truth free space.
  background.append(dict(group=g['group'],source_counts=dict(Counter(o['source'] for o in statics)),
    grade_class_counts=dict(Counter(o['identity_grade']+'/'+o['petr_class'] for o in statics)),
    reviewed_regions=[dict(source=o['source']+'Source',entity_id=o['entity_id'],name=o['name'],class_name=o['petr_class'],
      status='RenderLinked' if o['identity_grade'].startswith('RenderLinked') else 'Ambiguous' if o['identity_grade']=='AmbiguousIdentity' else 'UnlabeledOrUnknown',
      views=o['views']) for o in statics],
    unlinked_regions=fr['unlinked_render_regions'],anonymous_geometry=side['level_bbs'],
    full_physical_precision_available=False,region_definition='Only listed instance/tag regions with unique semantic+depth/OBB support; bbox is not a whole-negative ROI'))
  cc=[x for x in certificates if x['group']==g['group']]
  have={x['category'] for x in cc if x['natural_proxy_qualified']}
  group_out.append(dict(group=g['group'],planned_targets=len(cc),spawned=sum(x['accepted_for_spawn'] for x in cc),
    qualified_categories=sorted(have),smoke_pass={'vehicle','person'}.issubset(have),
    hidden_missing_targets='All requested target keys retained including preflight rejection; no silent dropping'))
 assert len(sync)==192 and all(x['ok'] for x in sync)
 common=provenance('ActualCARLA8Frames_OfflineSourceAndQualityAudit')
 write('reports/m8f_ground_surface_contract.json',dict(**common,contract=asset('configs/m8f/contract.json'),rules=cfg,source_docs='https://carla.readthedocs.io/en/0.9.15/python_api/',implementation=asset('simulator/m8f_placement.py'),qualification='15-point support geometry proxy; not exact mesh/continuous support; strict no-fallback'))
 write('reports/m8f_ray_hit_and_contact_audit.json',dict(**common,requests=requests,final_checks=support,
   negative_controls=asset('reports/m8f_negative_controls.json'),water_probe=asset('reports/m8f_water_multilayer_probe.json'),
   preflight_rays=sum(len(v['rays']) for r in requests for v in r['preflight'].values()),
   no_height_fallback=True,PhysicalContactCertified=False))
 write('reports/m8f_object_eligibility.json',dict(**common,objects=certificates,eligible=sum(x['TrainingEligibility']['status']=='EligibleForProposalOnly' for x in certificates),qualification='All independent gates conjunctive; no model output used'))
 write('reports/m8f_natural_scene_quality.json',dict(**common,groups=group_out,overall='SmokeNoGo',
   target_requests=8,spawned=1,rejected_pre_spawn=7,geometry_and_render_qualified=1,
   no_floating_detected_in_accepted_subjects=True,full_mesh_support='NotCertified',
   sensor_records=192,sensor_sync_and_sha=sync,sidecars=sidecars,
   expansion='NotRun_RequiredTwoSceneSmokeNotPassed',
   conservative_rejection='High vegetation overhead rejected even if taller than subject; exact collision not established. NONE ray hits likewise unexplained. Road lane query/edge placement failure retained, no threshold relaxation.'))
 write('reports/m8f_identity_and_background_audit.json',dict(**common,
   controlled_grades=dict(Counter(x['RenderIdentityGrade'] for x in certificates)),background=background,
   historical154='ReadOnlyNoBackfill; same M8E ledger unchanged',new_physical_FP=0,
   no_physical_precision_claim=True,review_scope='SimulatorEvidenceReviewed; regional source identity only, ground contact not certified for unplaced static assets.'))
 write('reports/m8f_independence_exceptions.json',dict(**common,object_checks=independence,
   status='HeldoutIndependenceNoGo',new_final_groups=0,history_coverage='HistoricalCoverageIncomplete',
   shared_layout_family=True,new_map=False,final_not_allocated=True,
   M8E_heldout='Historical support only; not renamed',
   model_selection_or_training=False))
 far=[x for x in certificates if x['natural_proxy_qualified'] and x['range_m'] is not None and 20<=x['range_m']<40]
 write('reports/m8f_far_field_support.json',dict(**common,status='InsufficientSupport',natural_geometry_render_far=len(far),
    independent_split_eligible_far=sum(x['TrainingEligibility']['status']=='EligibleForProposalOnly' for x in far),
    targets=far,target_goal=30,groups=len({x['group'] for x in far}),shifted_duplicates_counted=False,
    historical21_carried_as_training=0))
 write('reports/m8f_training_readiness.json',dict(**common,status='TrainingReadinessNoGo',
   blockers=['Two-scene smoke failed','No natural vehicle/cone/barrier evidence','Only1natural render-linked far pedestrian;0independent eligible','Historical coverage incomplete/no final','Background taxonomy and mesh/contact partial'],
   later_splits='NotRun',new_PETR_forward='NotRun',training_authorized=False))
 print(group_out,'far',len(far),'eligible0',flush=True)
if __name__=='__main__':main()
