"""One native spawn inventory and exactly 60 navigation calls; read-only geometry."""
import json,time
from collections import Counter
from pathlib import Path
import numpy as np
import yaml
from tools.m8h0_common import *
from tools.m8g_history import preflight
from datasets.m8f_quality import choose_surface
from datasets.m8h0_geometry import make_box,footprint,visibility
from simulator.m8f_placement import pose,box_record,ray,lane_check

def wp_record(wp):
    if wp is None:return None
    return dict(road_id=wp.road_id,lane_id=wp.lane_id,section_id=wp.section_id,s=wp.s,lane_type=str(wp.lane_type),lane_width_m=wp.lane_width,is_junction=wp.is_junction,transform=pose(wp.transform))

def audit(world,mp,transform,wp,asset,contract,environment):
    from simulator.m8f_placement import footprint as bottom_points,static_overlap
    g=contract['ground'];reg=contract['class_regions'][asset['category']];initial=make_box(transform.get_matrix(),asset);points=bottom_points(transform,asset['local_box_matrix'],asset['extent'],g['footprint_grid'])
    expected_z=wp.transform.location.z if wp is not None else transform.location.z
    rays=[];lanes=[];reasons=[];heights=[]
    for p in points:
        rr=ray(world,[p[0],p[1],expected_z+g['ray_above_m']],[p[0],p[1],expected_z-g['ray_below_m']]);d=choose_surface(rr['hits'],expected_z,reg['semantics'],g);rr['decision']=d;rays.append(rr)
        ln=lane_check(mp,[float(p[0]),float(p[1]),expected_z],reg['lanes'],g['lane_inset_m']);lanes.append(ln)
        if not ln['ok']:reasons.append('FootprintOutsideAllowedRegion')
        if d['status']!='GroundedByGeometryProxy':reasons.extend(d['reasons'])
        if d['selected'] is not None:heights.append(d['selected']['xyz'][2])
    if wp is None or str(wp.lane_type) not in reg['lanes']:reasons.append('DeclaredRegionUnsupported')
    if wp is not None:
        yawdelta=abs((transform.rotation.yaw-wp.transform.rotation.yaw+180)%360-180)
        if yawdelta>g['yaw_to_lane_max_deg']:reasons.append('NativeHeadingDisagreesWithDeclaredLane')
    else:yawdelta=None
    hypothetical=None;gaps=None;initial_gaps=None;competitors=[];matrix=None
    if len(heights)==len(points):
        initial_gaps=(points[:,2]-heights).tolist();delta=float(np.median(np.array(heights)-points[:,2])+g['requested_clearance_m']);m=np.array(transform.get_matrix());m[2,3]+=delta
        hypothetical=make_box(m,asset);matrix=m.tolist();gaps=(points[:,2]+delta-heights).tolist()
        if max(heights)-min(heights)>g['support_height_spread_m']:reasons.append('SupportHeightDisagreement')
        if min(gaps)<g['bottom_gap_min_m'] or max(gaps)>g['bottom_gap_max_m']:reasons.append('BottomSupportGap')
        competitors=static_overlap(hypothetical,environment)
        if competitors:reasons.append('StaticGeometryOverlap')
    else:reasons.append('MissingGroundEvidence')
    return dict(status='GeometryCandidateProxySupported' if not reasons else 'TemplateGeometryPartial',reasons=sorted(set(reasons)),rays=rays,lane_checks=lanes,expected_map_z_m=expected_z,native_initial_box=initial,native_initial_support_gaps_m=initial_gaps,native_z_offset_to_waypoint_m=transform.location.z-expected_z,heading_delta_deg=yawdelta,hypothetical_vertical_only_box=hypothetical,hypothetical_actor_matrix=matrix,hypothetical_support_gaps_m=gaps,static_competitors=competitors,unknowns=['Actor final settling and contact NotRun','Exact mesh and occluded space unknown','Render identity and background annotation NotRun'],not_a_spawn_authorization=True,native_xy_yaw_unchanged=True,original_surface_and_lane_rules_unchanged=True)

def main():
    import carla
    cfg=json.loads(Path('configs/m8h0/protocol.json').read_text());freeze=json.loads((ROOT/'m8g_freeze.json').read_text());history=json.loads(HISTORY.read_text());assert freeze['history']['sha256']==sha(HISTORY)
    client=carla.Client('127.0.0.1',2100);client.set_timeout(60);world=client.get_world();mp=world.get_map();assert mp.name.endswith(cfg['map']);assert client.get_server_version()==cfg['server_version_required']
    assert sha(ROOT/'Town02.xodr')==__import__('hashlib').sha256(mp.to_opendrive().encode()).hexdigest()
    original_assets=json.loads(Path('configs/m8f/protocol.json').read_text())['groups'][0]['objects'];assets={r['asset']['category']:r['asset'] for r in original_assets};contract=json.loads(Path('configs/m8f/contract.json').read_text());rig=yaml.safe_load(Path('configs/cameras.yaml').read_text())
    inventory=[dict(entity_id=int(e.id),name=e.name,semantic_tags=[int(e.type)],**box_record(e.bounding_box)) for e in world.get_environment_objects(carla.CityObjectLabel.Any)]
    actors=[dict(entity_id=a.id,name=a.type_id,semantic_tags=list(a.semantic_tags),**box_record(a.bounding_box,a.get_transform())) for a in world.get_actors() if not a.type_id.startswith(('sensor.','spectator','controller.'))]
    write(ROOT/'environment_inventory.json',dict(map=mp.name,world_episode=world.id,objects=inventory,actors=actors,reference='ActualCARLAEnvironmentAndActorOBBs_NotMesh',frame=world.get_snapshot().frame))
    bp=world.get_blueprint_library();available=[b.id for b in bp.filter('vehicle.*')];assert bp.find(cfg['vehicle_blueprint']) and bp.find(cfg['person_blueprint'])
    native=mp.get_spawn_points() # Exactly one inventory call. Preserve all native points even if over budget.
    write(ROOT/'native_spawn_points_raw.json',dict(**provenance('ActualMapCreatorSpawnPointInventory'),map=mp.name,map_xodr=asset(ROOT/'Town02.xodr'),world_episode=world.id,frame=world.get_snapshot().frame,records=[dict(index=i,transform=pose(t)) for i,t in enumerate(native)],count=len(native)))
    nav=[]
    with (ROOT/'navigation_returns.jsonl').open('x') as f:
        for i in range(cfg['navigation_requests']):
            snap=world.get_snapshot();point=world.get_random_location_from_navigation();row=dict(request_order=i,frame_before=snap.frame,timestamp_s=snap.timestamp.elapsed_seconds,world_episode=world.id,map=mp.name,xyz=None if point is None else [point.x,point.y,point.z]);f.write(json.dumps(row)+'\n');f.flush();nav.append(row)
    common=dict(map=mp.name,map_xodr=asset(ROOT/'Town02.xodr'),world_episode=world.id,history=asset(HISTORY),source_metadata='CARLA0.9.15 exact XODR; static OBB inventory SHA; packaged mesh/navmesh version not independently hashed')
    vehicles=[];people=[]
    for i,t in enumerate(native):
        wp=mp.get_waypoint(t.location,project_to_road=True,lane_type=carla.LaneType.Driving|carla.LaneType.Parking)
        if i>=cfg['max_vehicle_geometry_queries']:
            vehicles.append(dict(id='spawn_%03d'%i,index=i,transform=pose(t),source='MapCreatorSpawnPoint',waypoint=wp_record(wp),geometry=dict(status='NotRun_Budget'),status='VehiclePlacementCandidate'));continue
        g=audit(world,mp,t,wp,assets['vehicle'],contract,inventory);box=g['hypothetical_vertical_only_box'] or g['native_initial_box'];hist=preflight(footprint(box),'Town02',history,[t.location.x,t.location.y])
        vehicles.append(dict(id='spawn_%03d'%i,index=i,transform=pose(t),source='MapCreatorSpawnPoint',status='VehiclePlacementCandidate',waypoint=wp_record(wp),asset=assets['vehicle'],geometry=g,historical_preflight=hist,independent_split_eligible=False))
        print('vehicle',i,g['status'],g['reasons'],flush=True)
    for r in nav:
        if r['xyz'] is None:people.append(dict(**r,id='nav_%02d'%r['request_order'],source='ServerNavigationReturn',geometry=dict(status='NullNavigationReturn',reasons=['NoPoint']),independent_split_eligible=False));continue
        point=carla.Location(*r['xyz']);wp=mp.get_waypoint(point,project_to_road=True,lane_type=carla.LaneType.Sidewalk)
        rotation=carla.Rotation() if wp is None else wp.transform.rotation;t=carla.Transform(point,rotation);g=audit(world,mp,t,wp,assets['person'],contract,inventory);box=g['hypothetical_vertical_only_box'] or g['native_initial_box'];hist=preflight(footprint(box),'Town02',history)
        people.append(dict(**r,id='nav_%02d'%r['request_order'],source='ServerNavigationReturn',transform=pose(t),orientation_source='Proposed nearest sidewalk yaw; API returns position only',waypoint=wp_record(wp),asset=assets['person'],geometry=g,historical_preflight=hist,independent_split_eligible=False))
        print('navigation',r['request_order'],g['status'],g['reasons'],flush=True)
    # Annotate native source observation support against same finite native Ego set.
    for target in vehicles+people:
        g=target['geometry'];box=g.get('hypothetical_vertical_only_box') or g.get('native_initial_box');support=[]
        if box:
            for ego in vehicles:
                if ego['id']==target['id'] or ego['geometry']['status']!='GeometryCandidateProxySupported':continue
                egom=ego['geometry']['hypothetical_actor_matrix'];distance=np.linalg.norm(np.array(box['center_world_m'])[:2]-np.array(egom)[:2,3])
                if not 20<=distance<=40:continue
                v=visibility(egom,box,rig)
                if v['frustum_supported']:support.append(dict(ego_spawn_id=ego['id'],**v))
        target['nominal_20_40m_observation_options']=support;target['potential_20_40m']=bool(support)
    write(ROOT/'vehicle_inventory_full.json',dict(**provenance('ActualCARLAReadOnlyVehicleInventory'),**common,records=vehicles,available_vehicle_blueprints=available,dimension_verified_blueprints=[assets['vehicle']],other_vehicle_dimensions='Unknown_NotMeasured_NoNewActors'))
    write(ROOT/'navigation_inventory_full.json',dict(**provenance('ActualCARLAReadOnlyNavigationInventory'),**common,records=people,requests=len(nav),raw_returns=asset(ROOT/'navigation_returns.jsonl'),seed_replay_guarantee=False,exact_positions_are_replay_contract=True))
    for label,rows,file in [('native_vehicle_spawn_inventory',vehicles,'vehicle_inventory_full.json'),('pedestrian_navigation_inventory',people,'navigation_inventory_full.json')]:
        write('reports/m8h0_'+label+'.json',dict(**provenance('NativeCandidateSourceReviewed'),**common,raw=asset(ROOT/file),count=len(rows),geometry_counts=dict(Counter(r['geometry']['status'] for r in rows)),reasons=dict(Counter(k for r in rows for k in r['geometry'].get('reasons',[]))),potential_20_40m=sum(r['potential_20_40m'] for r in rows),records=[dict(id=r['id'],source=r['source'],transform=r.get('transform'),waypoint=r.get('waypoint'),geometry_status=r['geometry']['status'],reasons=r['geometry'].get('reasons',[]),initial_z_offset_m=r['geometry'].get('native_z_offset_to_waypoint_m'),initial_support_gaps_m=r['geometry'].get('native_initial_support_gaps_m'),observation_options=r['nominal_20_40m_observation_options'],historical_preflight=r.get('historical_preflight'),source_row_pointer=str(ROOT/file)+'#'+r['id']) for r in rows],source_status='NativeCandidateSourceReviewed',actual_settling='NotRun',render_identity='NotRun',no_candidate_is_approved_natural_target=True))
    print('COMPLETE',len(vehicles),len(people),'no actors/images',flush=True)
if __name__=='__main__':main()
