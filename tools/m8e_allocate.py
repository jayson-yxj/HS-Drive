"""Pre-render geometry-only positions, split separation and quota freeze."""
import json,math
from pathlib import Path
import numpy as np
from tools.m8e_common import *

def main():
    import carla
    client=carla.Client('127.0.0.1',2100);client.set_timeout(120)
    historical=[]
    for p in Path('configs').glob('*.json'):
        try:d=json.loads(p.read_text())
        except ValueError:continue
        if isinstance(d,dict):
            for g in d.get('groups',[]):
                if isinstance(g,dict) and 'pose' in g and 'map' in g:
                    historical.append(dict(map=g['map'].split('/')[-1],pose=g['pose'],source=str(p)))
    plan=[('development','Town04',3),('validation','Town03',3),('calibration','Town05',2),('held_out_final','Town01',2)]
    groups=[];allocation=[]
    for split,town,count in plan:
        world=client.load_world(town);mp=world.get_map()
        old=[np.asarray(x['pose'][:2]) for x in historical if x['map']==town]
        candidates=[]
        for wp in mp.generate_waypoints(10):
            if wp.lane_type!=carla.LaneType.Driving:continue
            t=wp.transform;xy=np.array([t.location.x,t.location.y])
            prior=min([np.linalg.norm(xy-o) for o in old] or [1e6])
            if prior<60:continue
            candidates.append((wp,xy,prior))
        chosen=[]
        for i in range(count):
            viable=[r for r in candidates if all(np.linalg.norm(r[1]-x[1])>=130 for x in chosen)]
            if not viable:break
            # No image/model selection; prefer larger historical spacing, straight area.
            row=max(viable,key=lambda r:(min(r[2],200)+20*(not r[0].is_junction),r[0].road_id,r[0].s))
            chosen.append(row);wp,xy,prior=row;t=wp.transform
            group_index=len(groups);rng=np.random.default_rng(88000+group_index)
            specs=[]
            # One distinct far target per each nominal camera direction.
            for j,bearing in enumerate([0,60,-60,120,-120,180]):
                kind=['vehicle.tesla.model3','vehicle.audi.tt','vehicle.mercedes.sprinter',
                      'walker.pedestrian.0001','static.prop.trafficcone01','static.prop.streetbarrier'][(j+group_index)%6]
                specs.append(dict(blueprint=kind,distance=float(rng.uniform(23,36)),bearing_hs_deg=float(bearing+rng.uniform(-8,8)),yaw_offset=float(rng.uniform(-100,100)),role='far'))
            # Distinct near/mid, overlap and occlusion stress; not repeated adjacent frames.
            for kind,d,b in [('vehicle.lincoln.mkz_2020',8,25),('walker.pedestrian.0002',14,-30),
                             ('static.prop.constructioncone',7,-115),('static.prop.streetbarrier',16,145)]:
                specs.append(dict(blueprint=kind,distance=d+float(rng.uniform(-1,1)),bearing_hs_deg=b+float(rng.uniform(-5,5)),yaw_offset=0.,role='near_mid_overlap'))
            groups.append(dict(group='m8e_%s_%02d'%(split,i),split=split,map=town,
                pose=[t.location.x,t.location.y,t.location.z+.4,t.rotation.pitch,t.rotation.yaw,t.rotation.roll],
                road_id=wp.road_id,lane_id=wp.lane_id,road_s=wp.s,is_junction=wp.is_junction,
                min_historical_ego_m=float(prior),weather=['ClearNoon','CloudyNoon','ClearSunset'][i%3],
                seed=88000+group_index,objects=specs,phases=['empty','present','shifted','removed'],
                shifted='same entities move bearing +12deg and radial +2m; correlated support, never independent count'))
        allocation.append(dict(map=town,split=split,requested=count,allocated=len(chosen)))
    protocol=dict(**provenance('PreCaptureGeometryOnlyAllocation'),groups=groups,allocation=allocation,
        limits=dict(max_groups=24,max_frames=96,planned_groups=len(groups),planned_frames=4*len(groups)),
        quotas=dict(far_unique_entity_position_observations=30,far_positions=6,all_six_cameras=True,
                    classes=['vehicle','person','cone','barrier'],multi_actor=True),
        isolation=dict(split_maps_disjoint=True,min_old_ego_distance_m=60,min_new_group_distance_m=130,
            historical_maps_reused=True,layout_families_shared=True,claim='New separated positions, not independent renderer/background distribution; target overlap audited after capture'),
        identity_rule=dict(min_pixels=3,min_retained_pixels=3,depth_range_m=[1.5,80],
            box_tolerance_m=.25,min_geometry_purity=.90,min_semantic_agreement=.95,
            unique_object_claim=True,actor_id_is_not_instance_id=True,
            source='Source-qualified entity + unique depth/OBB link; spawned identity additionally tests absent/present/removed',
            prediction_association='Existing score0.1 class-compatible one-to-one projected OBB IoU>=.1 proposal; same fixed M8C IoU>=.5/maskrecall>=.8/purity>=.9, now one valid view allowed with source link. No metric center error in identity.'),
        inference=dict(order='A',score=.1,held_out_final=False,optional_only_after_annotation_gate=True),
        no_training=True,no_closed_loop=True)
    write('configs/m8e/protocol.json',protocol)
    write('reports/m8e_scene_and_split_manifest.json',dict(protocol=protocol,historical_pose_sources=historical))
    print(allocation,'frames',4*len(groups))
if __name__=='__main__':main()
