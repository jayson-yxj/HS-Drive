"""Value-only temporal contracts. Planner cannot receive episode paths/future."""
from dataclasses import dataclass,asdict
import math
from datasets.m7a_contracts import Stamp,Box,OracleSceneState,PlanningRequest,WORLD


@dataclass(frozen=True)
class TemporalOracleEpisode:
 episode_id:str
 carla_version:str
 map_name:str
 map_sha256:str
 source:str
 frame_ids:tuple
 timestamps:tuple
 missing_frames:tuple


@dataclass(frozen=True)
class TemporalPlanningInput:
 episode_id:str
 frame:int
 simulation_timestamp_s:float
 input_available_wall_s:float
 signed_speed_mps:float
 ego_xy_yaw:tuple
 ego_z:float
 body_half_extent:tuple
 body_offset:tuple
 actors:tuple  # immutable Box values, no actor refs
 actor_metadata:tuple # id,category,role,blueprint only
 ego_route:tuple
 history_stamps:tuple
 source:str='Privileged_ActualSnapshot_OfflineOnly'
 version:str='m7d.v1'
 def validate(self,previous=None):
  if self.source!='Privileged_ActualSnapshot_OfflineOnly' or self.version!='m7d.v1':raise ValueError('source_or_version')
  if not all(math.isfinite(v) for v in [self.simulation_timestamp_s,self.input_available_wall_s,self.signed_speed_mps,*self.ego_xy_yaw,self.ego_z]):raise ValueError('nonfinite_input')
  if type(self.frame)is not int or self.frame<0:raise ValueError('invalid_frame')
  if len({x.object_id for x in self.actors})!=len(self.actors):raise ValueError('duplicate_actor_association')
  if any(t>self.simulation_timestamp_s for _,t in self.history_stamps):raise ValueError('future_in_history')
  for x in self.actors:x.validate()
  if previous:
   if previous.episode_id!=self.episode_id or self.frame<=previous.frame or self.simulation_timestamp_s<=previous.simulation_timestamp_s:raise ValueError('nonmonotonic_or_wrong_episode')
   dt=self.simulation_timestamp_s-previous.simulation_timestamp_s
   if abs(dt-(self.frame-previous.frame)*.05)>1e-5:raise ValueError('frame_time_mismatch')
   old={b.object_id:b for b in previous.actors}
   for b in self.actors:
    if b.object_id in old and math.dist(b.xy,old[b.object_id].xy)>max(2.,20*dt):raise ValueError('actor_association_jump')
   metadata={x[0]:x[1:] for x in previous.actor_metadata}
   if any(x[0] in metadata and metadata[x[0]]!=x[1:] for x in self.actor_metadata):raise ValueError('actor_identity_changed')
  if self.signed_speed_mps<-.05:raise ValueError('reverse_out_of_forward_planner_support')
 def request(self):
  self.validate();st=Stamp(self.frame,self.simulation_timestamp_s,self.simulation_timestamp_s+.2,WORLD)
  scene=OracleSceneState(st,(*self.ego_xy_yaw,max(0.,self.signed_speed_mps)),self.actors,self.body_half_extent,self.body_offset)
  return PlanningRequest(scene,self.ego_route,self.episode_id+':'+str(self.frame),self.ego_route[-1])


@dataclass(frozen=True)
class TemporalFutureReference:
 episode_id:str
 origin_frame:int
 origin_timestamp:float
 samples:tuple
 source:str='EvaluatorOnly_RecordedActualCARLA'


@dataclass(frozen=True)
class TemporalPlanningResult:
 episode_id:str
 input_frame:int
 input_timestamp_s:float
 input_available_wall_s:float
 compute_start_wall_s:float
 result_available_wall_s:float
 compute_duration_s:float
 modeled_result_sim_time_s:float
 expired_for_online:bool
 selected:str|None
 source:str='Privileged_OfflineShadow_NotControl'


def result_clock(inp,start,end,selected,input_delay_s=0.):
 duration=end-start
 if not math.isfinite(duration) or duration<0:raise ValueError('invalid_wall_clock')
 modeled=inp.simulation_timestamp_s+input_delay_s+duration
 return TemporalPlanningResult(inp.episode_id,inp.frame,inp.simulation_timestamp_s,inp.input_available_wall_s,start,end,duration,modeled,modeled>inp.simulation_timestamp_s+.2,selected)


def from_packet(p):
 allowed={'episode_id','frame','sim_time','input_available_wall_s','ego','body','actors','route','history_stamps','source','valid','missing'}
 if set(p)!=allowed or p['source']!='Privileged_ActualCARLAActorSnapshot' or not p['valid'] or p['missing']:raise ValueError('missing_snapshot_or_forbidden_fields')
 e=p['ego'];bs=[];metadata=[]
 for a in p['actors']:
  if a['frame']!=p['frame'] or abs(a['timestamp']-p['sim_time'])>1e-6:raise ValueError('actor_snapshot_stamp_mismatch')
  bs.append(Box(str(a['actor_id']),tuple(a['xy']),a['yaw'],a['half_length'],a['half_width'],tuple(a['velocity']),a.get('yaw_rate',0.)))
  metadata.append((str(a['actor_id']),a.get('category','Unknown'),a.get('role','Unknown'),a.get('blueprint','Unknown')))
 return TemporalPlanningInput(p['episode_id'],p['frame'],p['sim_time'],p['input_available_wall_s'],e['signed_speed'],(*e['xy'],e['yaw']),e['z'],tuple(p['body']['half_extent']),tuple(p['body']['offset']),tuple(bs),tuple(metadata),tuple(map(tuple,p['route'])),tuple(map(tuple,p['history_stamps'])))
