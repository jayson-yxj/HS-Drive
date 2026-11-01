"""Early byte routing, two-slot ownership and CPU-only partial preparation."""
import json,threading,time,traceback
from pathlib import Path
from queue import Queue,Empty
from datasets.m8j12_contract import CAMERAS
from tools.m8j13_common import write,append
from simulator.m8j13_partial_transport import PartialSlots

class EarlyRelay:
    """Drain callback queue off the tick owner; no JPEG in this process."""
    def __init__(self,inbox,slots,identity,root):
        self.source=inbox;self.slots=slots;self.identity=identity;self.root=root
        self.queue=Queue(96);self.lock=threading.RLock();self.frames={};self.tokens={};self.names={};self.closed=threading.Event();self.error=None
        self.thread=threading.Thread(target=self.run,name='camera-byte-relay',daemon=True);self.thread.start()
    def reserve_frame(self,frame,wall,index):
        with self.lock:self.frames[frame]=dict(wall=wall,index=index,status='AwaitingFirstCamera')
    def run(self):
        try:
            while not self.closed.is_set():
                for payload in self.source.drain():
                    stamp=payload.stamp.record();frame=stamp['frame'];start=time.monotonic()
                    with self.lock:
                        item=self.frames.get(frame)
                        if item is not None:
                            if item['status']=='AwaitingFirstCamera':
                                name='%06d.json'%item['index'];token=self.slots.begin(name,self.identity,frame,stamp['timestamp_s'],item['wall'])
                                if token is None:item['status']='QueueCapacity'
                                else:
                                    item.update(status='Partial',token=token,name=name);self.tokens[frame]=token;self.names[name]=frame
                            if item['status']=='Partial':self.slots.publish_camera(item['token'],stamp,payload.bgra)
                    self.queue.put_nowait(payload)
                    append(self.root/'relay_timing.jsonl',dict(frame=frame,camera=stamp['camera'],start_wall_s=start,done_wall_s=time.monotonic(),copied_wall_s=payload.copied_monotonic_s))
                self.closed.wait(.001)
        except BaseException as exc:self.error=repr(exc)
    @property
    def overflow(self):
        if self.error:raise RuntimeError('EarlyRelayFailed:'+self.error)
        return self.source.overflow
    def drain(self):
        if self.error:raise RuntimeError('EarlyRelayFailed:'+self.error)
        while True:
            try:yield self.queue.get_nowait()
            except Empty:return
    def commit(self,frame,row,snapshot,wall,sim):
        with self.lock:
            item=self.frames[frame]
            if item['status']!='Partial':raise RuntimeError('NotPublishablePartial:'+item['status'])
            self.slots.commit(item['token'],row,snapshot,wall,sim);item['status']='Committed';return item['name']
    def reject(self,frame):
        with self.lock:
            item=self.frames[frame]
            if item['status']=='Partial':self.slots.cancel(item['token']);item['status']='Cancelled'
            elif item['status']=='AwaitingFirstCamera':item['status']='RejectedBeforeAnyCamera'
    def release(self,name):
        with self.lock:
            frame=self.names[name];item=self.frames[frame];self.slots.release(name);self.tokens.pop(frame,None);item['status']='Released'
    def close(self):
        self.source.close();self.closed.set();self.thread.join(3)
        if self.thread.is_alive():raise RuntimeError('RelayExitTimeout')


def camera_prepare_loop(root,model_cfg,camera,pending,stop,age):
    from tools.m8j13_camera_preprocess import CameraPreprocessor
    prep=CameraPreprocessor(model_cfg,camera);slots=PartialSlots(json.loads((root/'transport.json').read_text())['root']);seen=set();identity=None
    try:
        while not stop.is_set() and not (root/'model_stop').exists():
            if not pending.reserve():continue
            headers=[]
            for p in slots.root.glob('*.partial.json'):
                h=json.loads(p.read_text());key=tuple(h['token'].values())
                if key not in seen:headers.append(h)
            if not headers:pending.cancel();stop.wait(.002);continue
            h=min(headers,key=lambda h:h['frame']);token=h['token'];seen.add(tuple(token.values()));name=token['request'];start=time.monotonic()
            if identity is None:identity=h['identity']
            if h['identity']!=identity:raise RuntimeError('CrossGenerationInput')
            prep.begin(token);submitted=set();aborted=False
            while not stop.is_set() and not (root/'model_stop').exists():
                h=json.loads((slots.root/('%d.partial.json'%token['slot'])).read_text())
                if h['token']!=token:raise RuntimeError('PartialLeaseOverwritten')
                if h['cancelled']:
                    aborted=True;break
                for name_cam in CAMERAS:
                    if name_cam in h['ready'] and name_cam not in submitted:
                        prep.submit(slots,token,name_cam);submitted.add(name_cam)
                if h['complete']:break
                stop.wait(.001)
            if stop.is_set() or (root/'model_stop').exists():prep.finish(token);pending.cancel();break
            if aborted:
                prep.finish(token);pending.publish(dict(cancelled=True,name=name,start=start));continue
            row=slots.committed_row(token);checked=age(row,start)
            # Original input start may predate CPU partial processing. Never renewed.
            item=dict(name=name,row=row,start=start,read_ms=0.,checked=checked,preprocess_start=start)
            write(root/'input_rows'/name,row)
            if checked['reason'] not in (None,'SimulationSnapshotUnconfirmed'):item['rejection']='InputRejectedBeforeAssembly'
            else:
                batch,timing=prep.assemble(slots,token);prepared=time.monotonic();item.update(cpu_batch=batch,prepared=prepared,preprocessing=timing,
                    images={n:prep.active[(token['slot'],token['generation'])][n].result()['original'] for n in CAMERAS},
                    prepare_thread_cpu_ms=timing['tail']['serial_tail_cpu_ms']+sum(x['cpu_ms'] for x in timing['cameras'].values()))
                item['checked']=age(row,start)
                if item['checked']['reason'] not in (None,'SimulationSnapshotUnconfirmed'):item['rejection']='InputRejectedAfterCPUPreprocess'
            prep.finish(token);pending.publish(item)
    except BaseException as exc:pending.publish(dict(error=repr(exc),traceback=traceback.format_exc()))
    finally:prep.close()
