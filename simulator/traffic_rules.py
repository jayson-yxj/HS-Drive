"""Privileged signal observations acquired between exclusive synchronous ticks.

CARLA's get_state is last-tick cached, not ActorSnapshot state. Read bracketing
frame IDs and mask two ticks after transitions for delayed camera rendering.
A green indication is never a full right-of-way / obstacle-free label.
"""
import numpy as np

SCHEMA='hsd-traffic-rules-v1'
STATES={'Red':0,'Yellow':1,'Green':2,'Off':3,'Unknown':4}


def xyz(v):return [v.x,v.y,v.z]


def signal_stop_target(state,*,applicable,stable):
    if not applicable or not stable or state not in ('Red','Green'):return -1,False
    return int(state=='Red'),True


def choose_probe(world):
    """Deterministic signal/approach with 12m of predecessor lane, no teleport in recording."""
    import carla
    for light in sorted(world.get_actors().filter('traffic.traffic_light'),key=lambda x:x.id):
        for stop in light.get_stop_waypoints():
            previous=stop.previous(12.)
            if len(previous)!=1 or previous[0].is_junction:continue
            transform=previous[0].transform
            transform.location.z+=.5
            return light,transform
    raise RuntimeError('No unambiguous signal approach available')


class TrafficRules:
    def __init__(self,world,*,probe=None):
        self.world=world;self.map=world.get_map();self.catalog=[];self.history={};self.probe=probe
        self.restoration=[];self.probe_group=[];self.original_global_frozen=None
        for light in sorted(world.get_actors().filter('traffic.traffic_light'),key=lambda a:a.id):
            stops=[]
            for wp in light.get_stop_waypoints():
                t=wp.transform
                stops.append({'road_id':wp.road_id,'section_id':wp.section_id,'lane_id':wp.lane_id,
                    'position_carla_world_m':xyz(t.location),'forward_carla_world':xyz(t.get_forward_vector()),
                    'right_carla_world':xyz(t.get_right_vector()),'lane_width_m':wp.lane_width,
                    'source':'CARLA get_stop_waypoints trigger-derived stop position, not painted-line annotation'})
            self.catalog.append((light,stops))
        if probe is not None:
            # 0.9.15 freeze() is global, not per-light. Snapshot all affected lights.
            self.restoration=[(light,light.get_state()) for light,_ in self.catalog]
            flags={light.is_frozen() for light,_ in self.catalog}
            if len(flags)!=1:raise RuntimeError('Cannot restore inconsistent global traffic freeze state')
            self.original_global_frozen=flags.pop();self.probe_group=probe.get_group_traffic_lights()
            try:
                probe.freeze(True);self.set_probe_phase(0)
            except BaseException:
                self.close();raise

    def set_probe_phase(self,tick):
        import carla
        # Controlled stationary diagnostic only: red -> yellow -> green -> red.
        phase=('Red','Yellow','Green','Red')[min(tick//20,3)]
        for light in self.probe_group:light.set_state(carla.TrafficLightState.Red)
        self.probe.set_state(getattr(carla.TrafficLightState,phase))

    def close(self):
        try:
            for light,state in self.restoration:light.set_state(state)
        finally:
            if self.probe is not None and self.original_global_frozen is not None:
                self.probe.freeze(self.original_global_frozen)
            self.restoration.clear()

    def capture(self,snapshot,ego):
        import carla
        frame=snapshot.frame
        if self.world.get_snapshot().frame!=frame:raise RuntimeError('Traffic label tick owner mismatch')
        pose=np.asarray(snapshot.find(ego.id).get_transform().get_matrix());inverse=np.linalg.inv(pose)
        wp=self.map.get_waypoint(carla.Location(*pose[:3,3]),project_to_road=False,lane_type=carla.LaneType.Driving)
        lane=None if wp is None else (wp.road_id,wp.section_id,wp.lane_id)
        signals=[];candidates=[]
        for light,stops in self.catalog:
            state=str(light.get_state());previous,changed=self.history.get(light.id,(state,frame))
            if state!=previous:changed=frame
            self.history[light.id]=(state,changed);stable=frame-changed>=2
            local=[]
            for stop in stops:
                point=np.array(stop['position_carla_world_m']);d=np.array(stop['forward_carla_world'])
                p=inverse[:3,:3]@(point-pose[:3,3]);p[1]*=-1
                forward=inverse[:3,:3]@d;forward[1]*=-1
                if np.linalg.norm(p)>85:continue
                right=inverse[:3,:3]@np.array(stop['right_carla_world']);right[1]*=-1
                ends=np.stack([p-right*stop['lane_width_m']/2,p+right*stop['lane_width_m']/2])
                applicable=bool(lane==(stop['road_id'],stop['section_id'],stop['lane_id']) and forward[0]>.8 and 0<=p[0]<=60 and abs(p[1])<stop['lane_width_m'])
                value,valid=signal_stop_target(state,applicable=applicable,stable=stable)
                local.append({**stop,'stop_center_hs_ego_m':p.tolist(),'stop_segment_hs_ego_m':ends.tolist(),
                    'forward_hs_ego':forward.tolist(),'applies_to_current_lane':applicable,'stop_required':value,'stop_valid':valid})
                if applicable:candidates.append((float(p[0]),light.id,state,value,valid))
            if local:signals.append({'actor_id':light.id,'state':state,'state_id':STATES.get(state,4),
                'stable_after_render_guard':stable,'stop_positions':local})
        if self.world.get_snapshot().frame!=frame:raise RuntimeError('Traffic labels straddled world ticks')
        candidates.sort();nearest=candidates[0] if candidates else None
        unique=nearest is not None and (len(candidates)==1 or candidates[1][0]-nearest[0]>.5)
        return {'schema_version':SCHEMA,'frame_id':frame,'timestamp_s':snapshot.timestamp.elapsed_seconds,
            'source':'last-tick cached TrafficLight state, bracketed by same-frame world reads',
            'controlled_probe':self.probe is not None,'transition_guard_ticks':2,'signals':signals,
            'ego_stop_required':nearest[3] if unique and nearest[4] else -1,
            'ego_stop_valid':bool(unique and nearest[4]),
            'ego_signal_state':STATES[nearest[2]] if unique and nearest[4] else -1,
            'ego_stop_distance_m':nearest[0] if unique else None,
            'full_traffic_permission_valid':False,
            'limitations':['no route choice','no yielding/priority','no obstacle clearance','yellow decision unknown','stop position comes from trigger, not painted line']}


def validate_traffic(record,frame,timestamp):
    recorded_time=record.get('timestamp_s',float('nan'))
    if (not np.isfinite(recorded_time) or not np.isfinite(timestamp) or
            record.get('schema_version')!=SCHEMA or record.get('frame_id')!=frame or abs(recorded_time-timestamp)>1e-5):
        raise ValueError('Traffic supervision frame/timestamp mismatch')
    if type(record['ego_stop_valid']) is not bool or type(record['full_traffic_permission_valid']) is not bool:raise ValueError('Rule validity must be boolean')
    if record['full_traffic_permission_valid']:raise ValueError('Incomplete traffic rules cannot assert full permission')
    if record['ego_stop_valid']:
        if record['ego_stop_required'] not in (0,1) or record['ego_signal_state'] not in (0,2):raise ValueError('Invalid signal stop target')
        if record['ego_stop_required']!=int(record['ego_signal_state']==STATES['Red']):raise ValueError('Ego signal truth table mismatch')
    elif record['ego_stop_required']!=-1 or record['ego_signal_state']!=-1:raise ValueError('Unknown stop target must remain unknown')

    for signal in record.get('signals',[]):
        if signal['state'] not in STATES or signal['state_id']!=STATES[signal['state']]:raise ValueError('Unknown/mismatched signal state')
        if type(signal['stable_after_render_guard']) is not bool:raise ValueError('Signal stability must be boolean')
        for stop in signal['stop_positions']:
            if type(stop['applies_to_current_lane']) is not bool or type(stop['stop_valid']) is not bool:raise ValueError('Signal masks must be boolean')
            segment=np.asarray(stop['stop_segment_hs_ego_m']);center=np.asarray(stop['stop_center_hs_ego_m'])
            if segment.shape!=(2,3) or center.shape!=(3,) or not np.isfinite(segment).all() or not np.isfinite(center).all():raise ValueError('Malformed stop geometry')
            expected=signal_stop_target(signal['state'],applicable=stop['applies_to_current_lane'],stable=signal['stable_after_render_guard'])
            if (stop['stop_required'],stop['stop_valid'])!=expected:raise ValueError('Signal truth table mismatch')
