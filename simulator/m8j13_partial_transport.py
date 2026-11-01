"""Opt-in per-camera publication on the existing two fixed mmap leases.

One publisher; immutable camera ranges. Only commit() marks a whole observation.
Reader CPU jobs never call CARLA. Creator releases only after all readers finish.
"""
import json,mmap,math
import numpy as np
from datasets.m8j12_contract import CAMERAS
from simulator.m8j13_transport import Slots,SIZE,binding
from tools.m8j13_common import atomic

class PartialSlots(Slots):
    def begin(self,request,identity,frame,timestamp,input_wall):
        if request in self.leases.values():raise RuntimeError('DuplicateRequest')
        slot=next((i for i in range(2) if i not in self.leases),None)
        if slot is None:return None
        if set(identity['sensors'])!=set(CAMERAS):raise ValueError('CameraIdentitySet')
        self.generations[slot]+=1;self.leases[slot]=request
        token=dict(slot=slot,generation=self.generations[slot],request=request)
        header=dict(token=token,identity=identity,frame=frame,timestamp=timestamp,input_wall=input_wall,ready=[],complete=False,cancelled=False)
        atomic(self.root/('%d.partial.json'%slot),header)
        return token
    def header(self,token):
        h=json.loads((self.root/('%d.partial.json'%token['slot'])).read_text())
        if h['token']!=token:raise RuntimeError('StalePartialLease')
        if h['cancelled']:raise RuntimeError('CancelledPartialLease')
        return h
    def publish_camera(self,token,stamp,bgra):
        h=self.header(token);name=stamp['camera'];identity=h['identity']
        if h['complete'] or name in h['ready']:raise RuntimeError('DuplicateCameraPublication')
        if name not in CAMERAS or stamp['actor_id']!=identity['sensors'][name]:raise ValueError('CameraInstanceMismatch')
        if (stamp['run'],stamp['episode'],stamp['sensor_generation'])!=(identity['run'],identity['episode'],identity['generation']):raise ValueError('GenerationMismatch')
        if stamp['frame']!=h['frame'] or not math.isfinite(stamp['timestamp_s']) or abs(stamp['timestamp_s']-h['timestamp'])>1e-5:raise ValueError('CameraClockMismatch')
        if len(bgra)!=SIZE//6:raise ValueError('CameraPayloadSize')
        index=CAMERAS.index(name);offset=index*(SIZE//6)
        with (self.root/('%d.raw'%token['slot'])).open('r+b') as f:
            with mmap.mmap(f.fileno(),SIZE,access=mmap.ACCESS_WRITE) as view:view[offset:offset+SIZE//6]=bgra
        record=dict(token=token,stamp=stamp,bytes=SIZE//6,shape=[396,704,4],dtype='uint8',layout='BGRA')
        atomic(self.root/('%d.%s.json'%(token['slot'],name)),record)
        h['ready'].append(name);atomic(self.root/('%d.partial.json'%token['slot']),h)
    def read_camera(self,token,name):
        h=self.header(token)
        if name not in h['ready']:raise RuntimeError('CameraNotPublished')
        p=self.root/('%d.%s.json'%(token['slot'],name));record=json.loads(p.read_text())
        if record['token']!=token:raise RuntimeError('StaleCameraLease')
        if (record['shape'],record['dtype'],record['layout'],record['bytes'])!=([396,704,4],'uint8','BGRA',SIZE//6):raise ValueError('CameraLayout')
        with (self.root/('%d.raw'%token['slot'])).open('rb') as f:
            with mmap.mmap(f.fileno(),SIZE,access=mmap.ACCESS_READ) as view:
                a=np.ndarray((396,704,4),np.uint8,buffer=view,offset=CAMERAS.index(name)*(SIZE//6));image=a[:,:,:3].copy();del a
        self.header(token)
        if json.loads(p.read_text())!=record:raise RuntimeError('CameraOverwrittenDuringRead')
        return image,record['stamp']
    def commit(self,token,row,snapshot,now_wall,now_sim):
        h=self.header(token)
        if h['complete'] or set(h['ready'])!=set(CAMERAS):raise RuntimeError('IncompleteOrDuplicateCommit')
        if now_wall-h['input_wall']>.25+1e-6:raise RuntimeError('InputWaitExpired')
        if now_sim-h['timestamp']>.3+1e-6:raise RuntimeError('ObservationSimulationExpired')
        if row['sequence_id']!=h['identity']['run'] or row['frame_id']!=h['frame'] or row['timestamp_s']!=h['timestamp'] or row['input_capture_wall_s']!=h['input_wall']:raise ValueError('RowBindingMismatch')
        if row['imu_frame']!=h['frame'] or abs(row['imu_timestamp_s']-h['timestamp'])>1e-5:raise ValueError('IMUClockMismatch')
        if snapshot['frame']!=h['frame'] or abs(snapshot['sim_s']-h['timestamp'])>1e-5 or snapshot['pose']!=row['ego_world_matrix'] or snapshot['speed']!=row['speed_mps']:raise ValueError('SnapshotBindingMismatch')
        for name in CAMERAS:
            rec=json.loads((self.root/('%d.%s.json'%(token['slot'],name))).read_text());s=rec['stamp']
            if rec['token']!=token or row['camera_frames'][name]!=s['frame'] or row['camera_timestamps_s'][name]!=s['timestamp_s'] or row['camera_world_matrices'][name]!=s['transform']:raise ValueError('RowCameraBindingMismatch')
        h.update(complete=True,row=row,row_binding=binding(row,h['identity']));atomic(self.root/('%d.partial.json'%token['slot']),h)
    def committed_row(self,token):
        h=self.header(token)
        if not h['complete']:raise RuntimeError('IncompleteObservation')
        if binding(h['row'],h['identity'])!=h['row_binding']:raise ValueError('RowBindingMismatch')
        return h['row']
    def cancel(self,token):
        h=self.header(token);h['cancelled']=True;atomic(self.root/('%d.partial.json'%token['slot']),h)
