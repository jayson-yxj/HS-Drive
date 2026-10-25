"""Source-qualified render/OBB association; does not assert exact physical mesh truth."""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from geometry.reference import unproject_pixels_to_optical,transform_points,OPTICAL_TO_CARLA_CAM,CARLA_EGO_TO_HS_EGO
from datasets.m8e_reference import points_inside,render_grade,coarse,require_source_frame
from datasets.m8b_petr_adapter import corners_hs
from tools.m8e_common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--groups',nargs='+',required=True);args=ap.parse_args()
    protocol=json.loads(Path('configs/m8e/protocol.json').read_text());cfg=protocol['identity_rule']
    out=ROOT/'audit';out.mkdir(exist_ok=True)
    for group in args.groups:
        directory=ROOT/'capture'/group;manifest=json.loads((directory/'manifest.json').read_text())
        target_ids=set();frames=[];allrows=[]
        for fref in manifest['frames']:
            s=json.loads(Path(fref['path']).read_text());target_ids.update(s['controlled_entity_ids'])
            require_source_frame(s)
            E=np.asarray(s['T_carla_world_from_ego']);ego=E[:3,3]
            env=json.loads(Path(s['environment_inventory']['path']).read_text())['objects']
            objects=[dict(o,is_controlled_target=o['entity_id'] in s['controlled_entity_ids']) for o in s['objects'] if not o.get('is_ego')]
            objects.extend(dict(o,dynamic_entity=False,frame_id=s['frame_id'],timestamp_s=s['timestamp_s']) for o in env)
            # All objects saved in inventory; only local finite-size entities eligible for link search.
            nearby=[o for o in objects if np.linalg.norm(np.array(o['center_world_m'])-ego)<90 and
                    min(o['dimensions_lwh_m'])>0 and max(o['dimensions_lwh_m'])<40]
            records={o['source']+':'+str(o['entity_id']):dict(o,views=[]) for o in nearby}
            unmatched=[];semantic_mismatch=[]
            for ci,cam in enumerate(s['calibration']):
                files=s['sensors'][cam['name']]
                with np.load(files['instance']['path'],allow_pickle=False) as z:ids=z['instance'];tags=z['semantic']
                with np.load(files['semantic']['path'],allow_pickle=False) as z:sem=z['semantic']
                with np.load(files['depth']['path'],allow_pickle=False) as z:depth=z['depth_m']
                semantic_mismatch.append(float(np.mean(tags!=sem)))
                combinations=np.unique(np.stack([tags.ravel(),ids.ravel()],axis=1),axis=0)
                for tag,rid in combinations:
                    mask=(ids==rid)&(tags==tag)&np.isfinite(depth)&(depth>=1.5)&(depth<80)
                    vv,uu=np.where(mask)
                    if len(vv)<3:continue
                    keep=vv>=151
                    points=unproject_pixels_to_optical(uu,vv,depth[vv,uu],np.array(cam['K']))
                    world=transform_points(points,np.asarray(cam['T_world_from_camera'])@OPTICAL_TO_CARLA_CAM)
                    center=world.mean(0);claims=[]
                    for key,o in records.items():
                        if o['semantic_tags'] and int(tag) not in o['semantic_tags']:continue
                        if not o['semantic_tags'] and not o.get('is_controlled_target'):continue
                        if np.linalg.norm(center-np.array(o['center_world_m']))>np.linalg.norm(o['extent_m'])+1:continue
                        inside=points_inside(world,o['T_world_from_box'],o['extent_m'],cfg['box_tolerance_m'])
                        purity=float(inside.mean())
                        if purity>=cfg['min_geometry_purity']:
                            claims.append((key,purity))
                    semagree=float(np.mean(sem[vv,uu]==tag))
                    accepted=len(claims)==1 and semagree>=cfg['min_semantic_agreement'] and keep.sum()>=3
                    record=dict(camera=cam['name'],camera_index=ci,frame_id=s['frame_id'],
                        render_encoding_id=int(rid),semantic_id=int(tag),native_pixels=len(vv),
                        retained_pixels=int(keep.sum()),depth_valid_pixels=len(vv),
                        retained_bbox=[int(uu[keep].min()),int(vv[keep].min()),int(uu[keep].max()+1),int(vv[keep].max()+1)] if keep.any() else None,
                        native_bbox=[int(uu.min()),int(vv.min()),int(uu.max()+1),int(vv.max()+1)],
                        truncated=bool((uu==0).any() or (uu==639).any() or (vv==0).any() or (vv==383).any()),
                        semantic_agreement=semagree,claims=[dict(entity_key=k,purity=p) for k,p in claims],
                        accepted=bool(accepted),identity_conflict=len(claims)>1,
                        linkage='UniqueSemanticAndDepthInsideOBB_NotIDEquality_NotMesh',
                        depth_median=float(np.median(depth[vv,uu])))
                    if len(claims)==1:
                        records[claims[0][0]]['views'].append(record)
                    elif len(claims)>1:
                        for key,purity in claims:records[key]['views'].append(dict(record,accepted=False))
                    else:unmatched.append(record)
            result=[]
            for key,o in records.items():
                if not o['views'] and not o.get('is_controlled_target'):continue
                center=transform_points([o['center_world_m']],CARLA_EGO_TO_HS_EGO@np.linalg.inv(E))[0]
                rotation=(CARLA_EGO_TO_HS_EGO@np.linalg.inv(E)@np.asarray(o['T_world_from_box']))[:3,:3]
                yaw=float(np.arctan2(rotation[1,0],rotation[0,0]))
                box=np.r_[center,o['dimensions_lwh_m'],yaw,0,0]
                exact=transform_points(o['corners_world_m'],CARLA_EGO_TO_HS_EGO@np.linalg.inv(E))
                approx=corners_hs(box)
                residual=float(np.linalg.norm(exact[:,None]-approx[None],axis=-1).min(1).max())
                result.append(dict(**o,entity_key=key,schema='ObjectReference.m8e.v1',group=group,split=s['split'],phase=s['phase'],
                    map=s['map'],world_episode=s['world_episode'],box_hs=box.tolist(),range_m=float(np.linalg.norm(center[:2])),
                    upright_corner_approximation_m=residual,identity_grade=render_grade(o['views']),
                    confidence='SimulatorEvidenceReviewed_NotHumanVerified_NotMeshTruth',
                    namespace_contract='Actor ID, EnvironmentObject uint64 ID, render encoding uint16 ID separate',
                    frame_consistent=bool(o.get('snapshot_present',True))))
            frame=dict(phase=s['phase'],frame_id=s['frame_id'],timestamp_s=s['timestamp_s'],sidecar=asset(fref['path']),
                objects=result,unlinked_render_regions=unmatched,semantic_sensor_mismatch_fraction=semantic_mismatch,
                local_reference_candidates=len(nearby),environment_total=len(env),actor_total=len(s['objects']))
            frames.append(frame);allrows.extend(result)
        # Cross-camera source identity and absent/present/removed control are distinct from mesh exactness.
        controls=[]
        for entity in sorted(target_ids):
            key='Actor:'+str(entity);states={}
            for frame in frames:
                row=next((r for r in frame['objects'] if r['entity_key']==key),None)
                states[frame['phase']]=dict(grade=row['identity_grade'] if row else 'UnlabeledOrUnknown',
                    native_pixels=sum(v['native_pixels'] for v in row['views'] if v['accepted']) if row else 0,
                    render_ids=sorted(set(v['render_encoding_id'] for v in row['views'] if v['accepted'])) if row else [])
            controls.append(dict(entity_key=key,states=states,
                absent_control_pass=states['empty']['native_pixels']==0 and states['removed']['native_pixels']==0))
        write(out/(group+'.json'),dict(**provenance('OfflineActualCARLAGeometryIdentityAudit'),group=group,frames=frames,
              controls=controls,rule=cfg,
              limits=['OBB encloses rendered evidence; not exact mesh or independent physical truth.',
                      'Multiple source boxes claim same semantic/instance region => AmbiguousIdentity; no nearest-ID guessing.',
                      'Unlinked regions retained Unknown, never safe Free. Static background completeness remains partial.']))
        present=[r for r in allrows if r['phase']=='present' and r.get('is_controlled_target')]
        print(group,Counter(r['identity_grade'] for r in present),'far',sum(20<=r['range_m']<40 and r['identity_grade'].startswith('RenderLinked') for r in present),flush=True)
if __name__=='__main__':main()
