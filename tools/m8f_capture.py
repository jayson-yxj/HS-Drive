"""Bounded static snapshot collection; no driving planner or visual control."""
import argparse,json,queue,time
from pathlib import Path
import numpy as np
import cv2,yaml
from simulator.session import CarlaSession
from geometry.reference import intrinsics_from_hfov,camera_mount_to_hs_from_optical,decode_carla_depth_bgra,CARLA_EGO_TO_HS_EGO,transform_points
from datasets.m8e_reference import taxonomy
from tools.m8f_common import *
from simulator.m8f_placement import validate_request,actual_check

def main():
    import carla
    ap=argparse.ArgumentParser();ap.add_argument('--groups',nargs='+',required=True);args=ap.parse_args()
    protocol=json.loads(Path('configs/m8f/protocol.json').read_text())
    rig=yaml.safe_load(Path('configs/cameras.yaml').read_text())
    cfg=yaml.safe_load(Path('configs/carla.yaml').read_text());cfg['port']=2100
    root=ROOT/'capture';root.mkdir(exist_ok=True)
    def tf(matrix):
        m=np.asarray(matrix);r=m[:3,:3]
        return carla.Transform(carla.Location(*m[:3,3]),carla.Rotation(
            pitch=float(np.degrees(np.arcsin(np.clip(r[2,0],-1,1)))),
            yaw=float(np.degrees(np.arctan2(r[1,0],r[0,0]))),
            roll=float(np.degrees(np.arctan2(-r[2,1],r[2,2])))))
    def bbox_record(bb,actor_tf=None):
        box_tf=carla.Transform(bb.location,bb.rotation)
        matrix=np.asarray(box_tf.get_matrix())
        if actor_tf is not None:matrix=np.asarray(actor_tf.get_matrix())@matrix
        ext=[bb.extent.x,bb.extent.y,bb.extent.z]
        corners=np.array([[x,y,z] for x in [-ext[0],ext[0]] for y in [-ext[1],ext[1]] for z in [-ext[2],ext[2]]])
        return dict(T_world_from_box=matrix.tolist(),center_world_m=matrix[:3,3].tolist(),
                    dimensions_lwh_m=(np.array(ext)*2).tolist(),extent_m=ext,
                    rotation_original_deg=[bb.rotation.pitch,bb.rotation.yaw,bb.rotation.roll],
                    corners_world_m=transform_points(corners,matrix).tolist())
    for g in protocol['groups']:
        if g['group'] not in args.groups:continue
        directory=root/g['group'];directory.mkdir(exist_ok=False)
        rgbdir=directory/'rgb';side=directory/'evaluator';rgbdir.mkdir();side.mkdir()
        cfg['map']=g['map'];q=queue.Queue()
        with CarlaSession(cfg,load_map=True) as session:
            world=session.world;world.set_weather(getattr(carla.WeatherParameters,g['weather']))
            p=g['ego'];spawn=carla.Transform(carla.Location(p['xyz'][0],p['xyz'][1],p['xyz'][2]+.5),carla.Rotation(pitch=p['rotation'][0],yaw=p['rotation'][1],roll=p['rotation'][2]))
            ego=session.spawn(world.get_blueprint_library().find('vehicle.tesla.model3'),spawn)
            ego.apply_control(carla.VehicleControl(brake=1,hand_brake=True))
            for _ in range(25):world.tick()
            ego.set_simulate_physics(False);pose=ego.get_transform();ego_matrix=np.asarray(pose.get_matrix())
            lib=world.get_blueprint_library()
            inventory=[]
            for e in world.get_environment_objects(carla.CityObjectLabel.Any):
                item=dict(source='EnvironmentObject',entity_id=int(e.id),name=e.name,semantic_tags=[int(e.type)],
                          entity_transform=e.transform.get_matrix(),**bbox_record(e.bounding_box))
                item['petr_class'],item['taxonomy_basis']=taxonomy(item['semantic_tags'],e.name)
                inventory.append(item)
            levels=[bbox_record(b) for b in world.get_level_bbs(carla.CityObjectLabel.Any)]
            write(side/'environment_inventory.json',dict(map=world.get_map().name,objects=inventory,
                  source='ActualCARLAEnvironmentObjects_NotMesh',query_frame=world.get_snapshot().frame))
            write(side/'level_bbs.json',dict(boxes=levels,source='AnonymousLevelBoundingBoxes_NoEntityID_NotMesh'))
            contract=json.loads(Path('configs/m8f/contract.json').read_text())
            requests=[];targets=[];selected=[];mp=world.get_map()
            for spec in g['objects']:
                req={}
                for phase,where in spec['phases'].items():
                    wp=mp.get_waypoint_xodr(where['road_id'],where['lane_id'],where['s'])
                    req[phase]=validate_request(world,mp,wp,spec['asset'],contract,inventory,where['offset_m'])
                accepted=all(r['status']=='GroundedByGeometryProxy' for r in req.values())
                requests.append(dict(key=spec['key'],preflight=req,accepted_for_spawn=accepted))
                if accepted:
                    # Legality established before creating this actual subject.
                    actor=session.spawn(lib.find(spec['asset']['blueprint']),tf(req['present']['request']['matrix']))
                    actor.set_simulate_physics(False)
                    actor.set_transform(carla.Transform(carla.Location(x=10000+len(targets)*10,y=10000,z=500)))
                    targets.append(actor);selected.append((spec,req))
            write(side/'grounded_requests.json',dict(**provenance('ActualCARLARayAndLanePreflight'),requests=requests))
            print(g['group'],'preflight',[(r['key'],r['accepted_for_spawn'],{k:v['reasons'] for k,v in r['preflight'].items()}) for r in requests],flush=True)
            calibration=[]
            for name,mount in rig['cameras'].items():
                local=carla.Transform(carla.Location(*mount['xyz_m']),carla.Rotation(yaw=mount['yaw_deg'],pitch=mount['pitch_deg']))
                transform=tf(ego_matrix@np.asarray(local.get_matrix()))
                calibration.append(dict(name=name,K=intrinsics_from_hfov(640,384,100).tolist(),
                    T_cam_to_ego=camera_mount_to_hs_from_optical(local.get_matrix()).tolist(),
                    T_world_from_camera=transform.get_matrix()))
                for kind in ['rgb','depth','instance_segmentation','semantic_segmentation']:
                    bp=lib.find('sensor.camera.'+kind)
                    for key,value in dict(image_size_x=640,image_size_y=384,fov=100,sensor_tick=0,lens_k=0,lens_kcube=0).items():
                        bp.set_attribute(key,str(value))
                    if kind=='rgb':bp.set_attribute('motion_blur_intensity','0')
                    sensor=session.spawn(bp,transform)
                    sensor.listen(lambda image,n=name,k=kind:q.put((n,k,image)))
            def tick():
                frame=world.tick();snap=world.get_snapshot();found={};deadline=time.monotonic()+45
                while len(found)<24:
                    if time.monotonic()>deadline:raise TimeoutError('Missing synchronized six RGB/Depth/Instance/Semantic')
                    n,k,img=q.get(timeout=10)
                    if img.frame==frame:found[n,k]=img
                    elif img.frame>frame:raise RuntimeError('Future sensor frame')
                return frame,snap,found
            for _ in range(12):
                world.tick();time.sleep(.06)
                while not q.empty():q.get_nowait()
            for _ in range(2):tick()
            frame_records=[]
            for phase in contract['phases']:
                placement_checks=[]
                for j,(actor,(spec,req)) in enumerate(zip(targets,selected)):
                    if phase in ['empty','removed']:
                        actor.set_transform(carla.Transform(carla.Location(x=10000+j*10,y=10000,z=500)))
                    else:
                        actor.set_transform(tf(req[phase]['request']['matrix']))
                for _ in range(4):tick()
                frame,snapshot,images=tick()
                if phase in ['present','shifted']:
                    for actor,(spec,req) in zip(targets,selected):
                        placement_checks.append(dict(entity_id=actor.id,key=spec['key'],category=spec['asset']['category'],
                            request=req[phase],final=actual_check(world,mp,actor,req[phase],contract,inventory)))
                write(side/(phase+'_support.json'),dict(group=g['group'],phase=phase,frame_id=frame,checks=placement_checks))
                packet=dict(schema='m8b.six_rgb.v1',token=g['group']+'_'+phase,camera_order=[c['name'] for c in calibration],
                            frame_id=frame,timestamp_s=snapshot.timestamp.elapsed_seconds,image_size=[384,640],cameras=[])
                objects=[]
                for actor in world.get_actors():
                    if actor.type_id.startswith(('sensor.','controller.','spectator')):continue
                    snap=snapshot.find(actor.id)
                    trans=snap.get_transform() if snap else actor.get_transform()
                    tags=list(actor.semantic_tags);petr,basis=taxonomy(tags,actor.type_id)
                    record=dict(source='Actor',entity_id=actor.id,name=actor.type_id,semantic_tags=tags,
                          entity_transform=trans.get_matrix(),snapshot_present=snap is not None,
                          frame_id=frame,timestamp_s=snapshot.timestamp.elapsed_seconds,
                          is_ego=actor.id==ego.id,is_controlled_target=actor.id in {a.id for a in targets},
                          dynamic_entity=actor.type_id.startswith(('vehicle.','walker.')),
                          petr_class=petr,taxonomy_basis=basis,**bbox_record(actor.bounding_box,trans))
                    objects.append(record)
                sensors={}
                for cam in calibration:
                    name=cam['name'];sensor_record={}
                    for kind,short in [('rgb','rgb'),('depth','depth'),('instance_segmentation','instance'),('semantic_segmentation','semantic')]:
                        image=images[name,kind];assert abs(image.timestamp-snapshot.timestamp.elapsed_seconds)<1e-5
                        arr=np.frombuffer(image.raw_data,np.uint8).reshape(384,640,4)
                        path=(rgbdir if short=='rgb' else side)/(phase+'_'+name+('.png' if short=='rgb' else '_'+short+'.npz'))
                        if path.exists():raise FileExistsError(path)
                        if short=='rgb':
                            ok=cv2.imwrite(str(path),arr[:,:,:3]);assert ok
                        else:
                            with path.open('xb') as stream:
                                if short=='depth':np.savez_compressed(stream,depth_m=decode_carla_depth_bgra(image.raw_data,384,640),raw_bgra=arr)
                                elif short=='instance':np.savez_compressed(stream,semantic=arr[:,:,2],instance=arr[:,:,1].astype(np.uint16)*256+arr[:,:,0].astype(np.uint16),raw_bgra=arr)
                                else:np.savez_compressed(stream,semantic=arr[:,:,2],raw_bgra=arr)
                        sensor_record[short]=dict(**asset(path),frame_id=image.frame,timestamp_s=image.timestamp,transform=image.transform.get_matrix())
                    sensors[name]=sensor_record
                    packet['cameras'].append(dict(name=name,path=str(rgbdir/(phase+'_'+name+'.png')),
                        sha256=sensors[name]['rgb']['sha256'],frame_id=frame,timestamp_s=snapshot.timestamp.elapsed_seconds,
                        K=cam['K'],T_cam_to_ego=cam['T_cam_to_ego']))
                pp=rgbdir/(phase+'_packet.json');write(pp,packet)
                sidecar=dict(schema='EvaluatorSidecar.m8f.v1',source='CARLA_Privileged_EvaluatorOnly',world_episode=world.id,
                    map=world.get_map().name,group=g['group'],split=g['split'],phase=phase,frame_id=frame,timestamp_s=snapshot.timestamp.elapsed_seconds,
                    T_carla_world_from_ego=ego_matrix.tolist(),calibration=calibration,objects=objects,sensors=sensors,
                    environment_inventory=asset(side/'environment_inventory.json'),level_bbs=asset(side/'level_bbs.json'),
                    ground_support=asset(side/(phase+'_support.json')),placement_requests=asset(side/'grounded_requests.json'),packet=asset(pp),controlled_entity_ids=[a.id for a in targets],source_sha=provenance('ActualCARLAStaticSnapshot'))
                sp=side/(phase+'_sidecar.json');write(sp,sidecar);frame_records.append(asset(sp))
                print(g['group'],phase,frame,len(objects),'objects',flush=True)
            write(directory/'manifest.json',dict(**provenance('ActualCARLAStaticCapture_NoDriving'),group=g,frames=frame_records,
                server=session.client.get_server_version(),all24sensors_synchronized=True,
                inventory_objects=len(inventory),level_boxes=len(levels),ego_physics_control_unchanged=True,static_actor_freeze_not_contact_evidence=True))
if __name__=='__main__':main()
