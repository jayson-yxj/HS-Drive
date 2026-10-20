"""Controlled near/far vehicle occlusion and reveal, using the calibrated six rig."""
import argparse,time,json
from pathlib import Path
from queue import Empty
import numpy as np
import cv2,yaml
from simulator.session import CarlaSession
from simulator.camera_rig import spawn_rig
from simulator.synchronizer import FrameSynchronizer
from simulator.snapshot import extract_snapshot
from simulator.recorder import Recorder,atomic_json
from datasets.schema import SENSOR_IDS
from geometry.reference import unproject_pixels_to_optical,transform_points
from geometry.carla_depth import decode_carla_depth_bgra,decode_rgb_bgra
from tools.provenance import source_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/carla.yaml');p.add_argument('--output',required=True)
    p.add_argument('--spawn-index',type=int,default=0);a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text());cam_cfg=yaml.safe_load(Path('configs/cameras.yaml').read_text())
    thresholds=yaml.safe_load(Path('configs/m2_acceptance.yaml').read_text())['occlusion']
    import carla
    rec=None;status='failed';sync=FrameSynchronizer(SENSOR_IDS)
    try:
        with CarlaSession(cfg) as session:
            world=session.world;bp=world.get_blueprint_library()
            base=world.get_map().get_spawn_points()[a.spawn_index]
            ego=session.spawn(bp.find('vehicle.tesla.model3'),base);ego.set_simulate_physics(False)
            def spawn_vehicle(name,x):
                pose=carla.Transform(base.transform(carla.Location(x=x,z=.05)),base.rotation)
                actor=session.spawn(bp.find(name),pose);actor.set_simulate_physics(False);return actor
            target=spawn_vehicle('vehicle.tesla.model3',16)
            blocker=spawn_vehicle('vehicle.carlamotors.carlacola',8)
            calibration=spawn_rig(session,ego,cam_cfg,sync)
            rec=Recorder(a.output,calibration,{'carla_version':'0.9.15','map':world.get_map().name,
                'camera_config':cam_cfg,'config':cfg,'source_sha256':source_hashes(),
                'capture_policy':'controlled_static_occlusion_reveal_physics_disabled',
                'target_id':target.id,'blocker_id':blocker.id})
            sync.on_event=rec.event
            def capture():
                # New phase, warm up the renderer and discard transition images.
                for _ in range(20):
                    world.tick();time.sleep(.3)
                    while True:
                        try:sync.queue.get_nowait()
                        except Empty:break
                sync.pending.clear();sync.snapshots.clear();sync.latest=-1
                sync.next_expected=None;sync.closed_through=-1
                for _ in range(30):
                    f=world.tick();snap=world.get_snapshot()
                    catalog={x.id:(x.type_id,x.bounding_box) for x in (ego,target,blocker) if x.is_alive}
                    state,objects=extract_snapshot(snap,ego,catalog)
                    sync.add_snapshot(f,snap.timestamp.elapsed_seconds,state,objects)
                    time.sleep(.3)
                    complete=sync.drain()
                    if complete:
                        rec.write(complete[-1]);return complete[-1]
                raise TimeoutError('No complete phase frame')
            blocked=capture()
            blocker.destroy();session.actors.remove(blocker)
            revealed=capture()
            camera=calibration['cameras']['front'];K=np.array(camera['K']);E=np.array(camera['T_hs_ego_from_opt'])
            def depth(bundle):
                im=bundle.sensors['front/depth'];return decode_carla_depth_bgra(im.raw_bgra,im.height,im.width)
            far=depth(revealed);near=depth(blocked)
            corners=np.array([o['corners_hs_ego_m'] for o in revealed.objects if o['actor_id']==target.id][0])
            lo,hi=corners.min(0),corners.max(0)
            v,u=np.indices(far.shape)
            points=transform_points(unproject_pixels_to_optical(u,v,far,K),E).reshape(*far.shape,3)
            target_mask=np.all((points>=lo-.05)&(points<=hi+.05),axis=-1)
            target_mask=cv2.erode(target_mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            # Independent target center patch must be visible after reveal.
            center=(lo+hi)/2;opt=transform_points([center],np.linalg.inv(E))[0]
            uv=K@opt;cx,cy=np.rint(uv[:2]/uv[2]).astype(int)
            core=np.zeros_like(target_mask);core[max(0,cy-3):cy+4,max(0,cx-3):cx+4]=True
            visible_fraction=float(np.mean(target_mask[core]))
            blocked_fraction=float(np.mean((near<far-.5)[target_mask])) if target_mask.any() else 0.
            panels=[]
            for bundle,label in ((blocked,'BLOCKED'),(revealed,'REVEALED')):
                im=bundle.sensors['front/rgb'];rgb=decode_rgb_bgra(im.raw_bgra,im.height,im.width)
                panel=rgb[...,::-1].copy();panel[target_mask]=(.5*panel[target_mask]+np.array([0,120,0])).astype(np.uint8)
                cv2.putText(panel,label,(12,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2);panels.append(panel)
            cv2.imwrite(str(Path(a.output)/'occlusion.png'),np.concatenate(panels,axis=1))
            np.savez_compressed(Path(a.output)/'occlusion_depth.npz',blocked_m=near,revealed_m=far,target_mask=target_mask)
            result={'map':world.get_map().name,'mode':'oracle_geometry','thresholds':thresholds,
                'blocked_frame':blocked.frame_id,'revealed_frame':revealed.frame_id,
                'target_surface_pixels':int(target_mask.sum()),'blocked_fraction':blocked_fraction,
                'revealed_core_visible_fraction':visible_fraction,
                'passed':int(target_mask.sum())>=thresholds['min_test_pixels'] and blocked_fraction>=thresholds['min_blocked_fraction'] and visible_fraction>=thresholds['min_visible_fraction'],
                'scope':'One controlled front-camera truck/vehicle occlusion; mask based on revealed depth within independent GT target box. No hidden target/free labels inferred.'}
            atomic_json(Path(a.output)/'occlusion_report.json',result);print(json.dumps(result,indent=2));status='complete'
    finally:
        if rec:rec.close(status,sync.stats)


if __name__=='__main__':main()
