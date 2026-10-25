"""Bounded actual physics settling; capture only after explicit geometry gates."""
import copy,json,math
from pathlib import Path
import numpy as np
import yaml
from tools.m8h1_common import *
from tools.m8h1_preflight import tf
from datasets.m8h1_settling import stationary,drift
from simulator.session import CarlaSession
from simulator.m8f_placement import pose,box_record,actual_check,xyz
from datasets.m8h0_geometry import pair_relation

class TemplateStop(RuntimeError):pass

def physics_record(actor):
    if not actor.type_id.startswith('vehicle.'):return dict(type='WalkerDefault_NoPhysicsControlMutation')
    pc=actor.get_physics_control()
    names=['mass','drag_coefficient','center_of_mass','use_gear_autobox','gear_switch_time','clutch_strength','final_ratio','max_rpm','moi','damping_rate_full_throttle','damping_rate_zero_throttle_clutch_engaged','damping_rate_zero_throttle_clutch_disengaged','use_sweep_wheel_collision']
    out={k:(xyz(getattr(pc,k)) if k=='center_of_mass' else getattr(pc,k)) for k in names if hasattr(pc,k)}
    out['wheels']=[{k:(xyz(getattr(w,k)) if k=='position' else getattr(w,k)) for k in ['tire_friction','damping_rate','max_steer_angle','radius','max_brake_torque','max_handbrake_torque','position'] if hasattr(w,k)} for w in pc.wheels]
    out['full_repr']=str(pc);out['changed']=False;return out

def settle(session,actor,initial,cfg,trace_path,tick=None):
    import carla
    world=session.world;trace=[];decision={};abort=None
    with trace_path.open('x') as f:
        for k in range(cfg['max_ticks']):
            if actor.type_id.startswith('vehicle.'):actor.apply_control(carla.VehicleControl(throttle=0,brake=1,hand_brake=True))
            else:actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(0,0,0),speed=0,jump=False))
            if tick is None:world.tick()
            else:tick()
            snap=world.get_snapshot();state=snap.find(actor.id)
            if state is None:abort='MissingActorSnapshot';break
            p=pose(state.get_transform());r=dict(tick=k,frame_id=snap.frame,timestamp_s=snap.timestamp.elapsed_seconds,actor_id=actor.id,pose=p,velocity_mps=xyz(state.get_velocity()),acceleration_mps2=xyz(state.get_acceleration()),angular_velocity_deg_s=xyz(state.get_angular_velocity()),control=str(actor.get_control()),physics_enabled=True);trace.append(r);f.write(json.dumps(r)+'\n');f.flush()
            d=drift(initial,p,cfg);tilt=max(abs((p['rotation'][j]+180)%360-180) for j in [0,2])
            if d['xy_m']>cfg['abort_xy_drift_m'] or tilt>cfg['abort_tilt_deg']:abort='AbnormalDriftOrTilt';break
            decision=stationary(trace,cfg)
            if decision['stable']:break
    final=pose(actor.get_transform());dd=drift(initial,final,cfg)
    return dict(status='SettledPoseObserved' if decision.get('stable') and dd['passed'] and not abort else 'SettlingNoGo',stationarity=decision,drift=dd,abort=abort,max_ticks=cfg['max_ticks'],ticks=len(trace),trace=asset(trace_path),final_pose=final,final_OBB=box_record(actor.bounding_box,actor.get_transform()),last_velocity_mps=xyz(actor.get_velocity()),physics_never_disabled=True)

def evaluate_final(session,actor,request,initial,contract,env,cfg):
    # Preserve full original static-request result. Natural vertical settling is
    # evaluated separately, with unchanged support/region thresholds.
    raw=actual_check(session.world,session.world.get_map(),actor,request,contract,env)
    observed=copy.deepcopy(request);observed['request']=pose(actor.get_transform())
    support=actual_check(session.world,session.world.get_map(),actor,observed,contract,env)
    d=drift(initial,observed['request'],cfg)
    asset_spec=request['asset'];local=box_record(actor.bounding_box);match=np.allclose(local['extent_m'],asset_spec['extent'],atol=1e-5) and np.allclose(local['T_world_from_box'],asset_spec['local_box_matrix'],atol=1e-5)
    complete=len(support['support_gaps_m'])==15
    reasons=list(support['reasons'])
    if not complete:reasons.append('Incomplete15PointSupport')
    if not d['passed']:reasons.append('ActualNativeXYOrRotationDrift')
    if not match:reasons.append('ActualLocalBoxContractMismatch')
    return dict(status='GroundedByGeometryProxy' if not reasons else 'GeometryNoGo',reasons=sorted(set(reasons)),original_static_request_result=raw,observed_pose_support_result=support,independent_native_drift=d,actual_local_box=local,local_box_matches_frozen=bool(match),all15_support_observed=complete,thresholds_unchanged=True,pose_anchor_scope='Observed settled pose only; no inference that original hypothetical pose was executed. Natural vertical displacement explicitly reported.',precise_mesh_contact_certified=False)

def spawn_and_review(session,group,role,phase,contract,env,protocol,directory):
    import carla
    entry=group['poses'][role];where=entry['phases'][phase];record=dict(role=role,phase=phase,source_id=where['source_id'],requested_spawn=where['spawn_transform'],preflight=where['preflight'],status='NotRun',spawn_success=False)
    path=directory/(role+'_'+phase+'_review.json');actor=None
    try:
        actor=session.spawn(session.world.get_blueprint_library().find(entry['asset']['blueprint']),tf(where['spawn_transform']));record.update(status='SpawnSucceeded',spawn_success=True,actor_id=actor.id,initial_pose=where['spawn_transform'],pre_tick_cached_pose=pose(actor.get_transform()),initial_pose_source='Frozen spawn RPC transform, not pre-tick cache',initial_OBB=box_record(actor.bounding_box,tf(where['spawn_transform'])),physics=physics_record(actor),local_box=box_record(actor.bounding_box))
        record['settling']=settle(session,actor,record['initial_pose'],protocol['settling'],directory/(role+'_'+phase+'_trace.jsonl'))
        record['geometry']=evaluate_final(session,actor,where['preflight'],record['initial_pose'],contract,env,protocol['settling'])
        record['status']='SettledGeometryProxyGo' if record['settling']['status']=='SettledPoseObserved' and record['geometry']['status']=='GroundedByGeometryProxy' else 'ActualActorGeometryNoGo'
    except Exception as e:record.update(status='ActualActorException',error=type(e).__name__+': '+str(e))
    write(path,dict(**provenance('ActualCARLAPhysicsActorReview'),**record));print(group['template'],role,record['status'],record.get('geometry',{}).get('reasons'),flush=True)
    return actor,record

def main():
    import carla
    protocol=json.loads(Path('configs/m8h1/protocol.json').read_text());preflight=json.loads(Path('reports/m8h1_shifted_contract.json').read_text());contract=json.loads(Path('configs/m8f/contract.json').read_text());env=json.loads((ROOT/'environment_inventory.json').read_text())['objects'];out=[]
    cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg.update(map='Town02',port=2100);root=ROOT/'capture';root.mkdir(exist_ok=True)
    for group in preflight['groups']:
        directory=root/group['template'];directory.mkdir(exist_ok=False);reviews={};row=dict(template=group['template'],status='NotRun',frames=[],reviews={},shifted_contract=group['status'])
        if group['status']!='ShiftedContractGo':row.update(status='ShiftedContractNoGo');write(directory/'result.json',row);out.append(row);continue
        try:
            with CarlaSession(cfg,load_map=False) as session:
                session.world.set_weather(carla.WeatherParameters.ClearNoon)
                for role in ['ego','vehicle','person']:
                    actor,review=spawn_and_review(session,group,role,'present',contract,env,protocol,directory);reviews[role]=(actor,review);row['reviews'][role]=asset(directory/(role+'_present_review.json'))
                    if review['status']!='SettledGeometryProxyGo':raise TemplateStop(role+' failed stable actual geometry; no downstream generation/capture')
                pairs=[]
                for a,b in [('ego','vehicle'),('ego','person'),('vehicle','person')]:
                    relation=pair_relation(reviews[a][1]['settling']['final_OBB'],reviews[b][1]['settling']['final_OBB']);pairs.append(dict(first=a,second=b,**relation))
                row['actual_pairs']=pairs
                if not all(r['pair_clearance_pass'] for r in pairs):raise TemplateStop('ActualActorPairOverlap')
                # Lazy import: sensor module is never invoked for failed prerequisites.
                from tools.m8h1_capture import capture
                row.update(capture(session,group,reviews,contract,env,protocol,directory))
        except TemplateStop as e:row.update(status='DevelopmentSmokeNoGo',stop_reason=str(e))
        except Exception as e:row.update(status='DevelopmentSmokePartial',stop_reason=type(e).__name__+': '+str(e))
        write(directory/'result.json',dict(**provenance('ActualBoundedTemplateAttempt'),**row));out.append(row)
    write(ROOT/'actor_smoke_results.json',dict(**provenance('ActualCARLABoundedActorSmoke'),templates=out,max_attempts=1,all_attempts_retained=True))
    print([(r['template'],r['status'],r.get('stop_reason')) for r in out],flush=True)
if __name__=='__main__':main()
