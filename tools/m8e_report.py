"""Aggregate source/data support, never turn annotation coverage into perception success."""
import json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from tools.m8e_common import *
from datasets.m8e_reference import SEMANTIC,taxonomy,coarse

def stats(x):
    a=np.asarray(x,float)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
                p95=float(np.percentile(a,95)) if len(a) else None,max=float(a.max()) if len(a) else None)

def main():
    p=json.loads(Path('configs/m8e/protocol.json').read_text())
    audits=[json.loads((ROOT/'audit'/(g['group']+'.json')).read_text()) for g in p['groups']]
    rows=[];frames=[];manifests=[]
    sync=[];packets=[];group_actual={}
    for group,audit in zip(p['groups'],audits):
        manifest=json.loads((ROOT/'capture'/group['group']/'manifest.json').read_text())
        manifests.append(asset(ROOT/'capture'/group['group']/'manifest.json'))
        for f in audit['frames']:
            frames.append(dict(group=group['group'],split=group['split'],**f))
            rows.extend(dict(r) for r in f['objects'])
            side=json.loads(Path(f['sidecar']['path']).read_text())
            packet=json.loads(Path(side['packet']['path']).read_text());packets.append(side['packet'])
            group_actual[group['group']]=np.asarray(side['T_carla_world_from_ego'])[:3,3].tolist()
            for camera in side['sensors'].values():
                for sensor in camera.values():
                    sync.append(sensor['frame_id']==side['frame_id'] and abs(sensor['timestamp_s']-side['timestamp_s'])<1e-5 and sha(sensor['path'])==sensor['sha256'])
            assert set(packet)=={'schema','token','camera_order','frame_id','timestamp_s','image_size','cameras'}
            assert all(set(c)=={'name','path','sha256','frame_id','timestamp_s','K','T_cam_to_ego'} for c in packet['cameras'])
    present=[r for r in rows if r['phase']=='present']
    controlled=[r for r in present if r.get('is_controlled_target')]
    linked=[r for r in controlled if r['identity_grade'].startswith('RenderLinked')]
    far=[r for r in linked if 20<=r['range_m']<40]
    # Source IDs scoped by episode/group; count one present state, not shifted duplicate.
    unique_far={(r['group'],r['entity_key']) for r in far}
    schema=dict(**provenance('ActualCARLACapture_OfflineEvaluatorAggregation'))
    write('reports/m8e_actor_environment_gt.json',dict(**schema,frames=len(frames),groups=len(audits),
        all24sensor_payloads_verified=all(sync),sensor_records=len(sync),manifests=manifests,RGB_only_packets=packets,
        source_objects_by_frame=[dict(group=f['group'],phase=f['phase'],actor_total=f['actor_total'],
            environment_total=f['environment_total'],nearby_candidates=f['local_reference_candidates'],
            source_linked_counts=dict(Counter((r['source']+'/'+r['identity_grade']) for r in f['objects']))) for f in frames],
        ObjectReference_contract='World/Map/frame/time, source/entity namespace, class, full box matrix/dimensions, yaw approximation, dynamic flag, per-camera raw instance evidence and depth/semantic checks in audit files.',
        audit_assets=[asset(ROOT/'audit'/(g['group']+'.json')) for g in p['groups']],
        bbox_limit='EnvironmentObject and Actor boxes are geometric envelopes, not exact render mesh. Anonymous get_level_bbs lacks entity identity.',
        semantic_sensor_disagreement=stats([x for f in frames for x in f['semantic_sensor_mismatch_fraction']])))
    source_grade=Counter(r['identity_grade'] for r in controlled)
    write('reports/m8e_instance_identity_audit.json',dict(**schema,controlled_present_targets=len(controlled),
        grades=dict(source_grade),controls=[dict(group=a['group'],**c) for a in audits for c in a['controls']],
        all_absent_controls_pass=all(c['absent_control_pass'] for a in audits for c in a['controls']),
        canonical_render_id='Raw BGRA: UE UniqueID low16 = G + 256*B. Saved legacy opaque render_encoding_id = 256*G+B. Never compared numerically to Actor.id or EnvironmentObject.id.',
        source_identity='Unique compatible depth points in OBB + semantic sensor agreement; unknown IDs or multiple claims never silently assigned.',
        collision_records=[dict(group=r['group'],phase=r['phase'],entity_key=r['entity_key'],views=r['views']) for r in rows if r['identity_grade']=='AmbiguousIdentity'],
        evidence_level='SimulatorEvidenceReviewed',human_double_verified=False,
        object_rows=controlled))
    far_groups=sorted({r['group'] for r in far})
    far_status='FarFieldSupportSufficientForProposalReview' if len(unique_far)>=30 and len(far_groups)>=6 else 'FarFieldSupportInsufficient'
    write('reports/m8e_far_field_support.json',dict(**schema,status=far_status,planned=p['quotas'],
        unique_present_far_targets=len(unique_far),far_groups=far_groups,far_group_count=len(far_groups),
        classes=dict(Counter(coarse(r['petr_class']) for r in far)),camera_views=dict(Counter(v['camera'] for r in far for v in r['views'] if v['accepted'])),
        qualifying_objects=[dict(group=r['group'],entity_key=r['entity_key'],type=r['petr_class'],range_m=r['range_m'],grade=r['identity_grade']) for r in far],
        shifted_frames_not_added_to_unique_count=True,over40_linked=sum(r['range_m']>=40 for r in linked),
        missing='No planned >40m target; do not infer beyond40 quality.',independence='Entities unique by group/episode, same assets and layout families correlated, not safety certification'))
    strata={}
    for name,selected in [('all',controlled),*[(s,[r for r in controlled if r['split']==s]) for s in ['development','validation','calibration','held_out_final']],
                          *[(c,[r for r in controlled if coarse(r['petr_class'])==c]) for c in ['vehicle','person','cone','barrier']]]:
        strata[name]=dict(targets=len(selected),grades=dict(Counter(r['identity_grade'] for r in selected)),
            accepted_views=dict(Counter(v['camera'] for r in selected for v in r['views'] if v['accepted'])))
    write('reports/m8e_multiview_identity.json',dict(**schema,strata=strata,
        single_and_multi_are_separate=True,merge_key='episode + group + source namespace + entity_id',
        views_not_independent_objects=True,contradictions='Any competing source link in a view yields AmbiguousIdentity; no cross-camera ID coincidence matching',
        source_only_no_detector_recall=True))
    mapping=[dict(semantic_id=i,name=n,default_class=taxonomy([i],'unknown')[0],basis=taxonomy([i],'unknown')[1]) for i,n in SEMANTIC.items()]
    write('reports/m8e_taxonomy_and_ignore_policy.json',dict(**schema,mapping=mapping,
        asset_overrides={'trafficcone/constructioncone':'traffic_cone','static.prop.streetbarrier':'barrier'},
        limitations=['GuardRail/Walls/Fences not automatically PETR barrier','Static/Dynamic alone cannot distinguish cone/barrier',
            'Car semantic does not fully specify subclasses; known blueprint names retained','Train/Rider/Other unsupported/unknown, not physical FP',
            'Ignore means not in supported detector taxonomy, never safety Free'],
        all10PETRclasses=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone'],
        unsupported_in_capture=['construction_vehicle','trailer','motorcycle','bicycle','bus']))
    # Real cross-split target separation, not merely ego distance.
    overlap=[];cross_dist=[]
    for i,g in enumerate(p['groups']):
        for h in p['groups'][i+1:]:
            if g['map']!=h['map'] or g['split']==h['split']:continue
            overlap.append([g['group'],h['group']])
    historic=json.loads(Path('reports/m8e_scene_and_split_manifest.json').read_text())['historical_pose_sources']
    geometry=[]
    for g in p['groups']:
        old=[np.asarray(h['pose'][:2]) for h in historic if h['map']==g['map']]
        c=[r for r in controlled if r['group']==g['group']]
        geometry.append(dict(group=g['group'],actual_ego_nearest_old_m=min([float(np.linalg.norm(np.array(group_actual[g['group']][:2])-o)) for o in old] or [1e6]),
            target_center_nearest_old_ego_m=min([float(np.linalg.norm(np.array(r['center_world_m'][:2])-o)) for r in c for o in old] or [1e6]),
            road_id=g['road_id'],new_map=False,shared_layout_family=True))
    write('reports/m8e_split_independence.json',dict(**schema,new_split_map_overlap=overlap,geometry=geometry,
        split_by_group=True,heldout_model_inference_calls=0,old_maps_reused=True,new_independent_distribution=False,
        min_planned_group_distance_m=130,limitations='Geometry-separated new locations on historical maps. Repeated blueprint/layout families and renderer remain correlated. No claim of independent map/domain test.'))
    write('reports/m8e_frozen_petr_diagnostic.json',dict(**schema,status='NotRun_AnnotationSupportGateNotMet',
        new_RGB_inference_calls=0,held_out_final_inference_calls=0,order_if_authorized='A',score=.1,
        reason='Incomplete static object identity and limited far/class support; optional GPU diagnostic not triggered.',
        historical_only=asset('reports/m8e_historical_evaluator.json')))
    write('reports/m8e_training_readiness.json',dict(**schema,status='TrainingReadinessNoGo',
        data_contract='DataContractGo' if all(sync) else 'NoGo',identity='IdentitySupportPartial',
        annotation='AnnotationCompletenessPartial',far=far_status,
        evaluator='LimitedEvaluatorContractReady_FullEvaluationBaselineNotReady',
        unresolved=['EnvironmentObject-to-render identity not complete','Source geometry is OBB not exact mesh',
            'Historical far model association remains1 only','New positions share historical maps/layout families',
            'Class/camera coverage insufficient for generalization','No automatic domain adaptation authorization'],
        training_authorized=False))
    print('groups',len(audits),'frames',len(frames),'controlled',len(controlled),'linked',len(linked),'far',len(unique_far),len(far_groups),dict(source_grade),flush=True)
if __name__=='__main__':main()
