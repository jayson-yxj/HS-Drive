"""Two fixed mmap slots. Owner publishes immutable lease; response releases it.

No multiprocessing resource tracker: creator unlinks only after readers exit.
Header is atomic sidecar; raw bytes never publish until the complete write ends.
"""
import hashlib,json,mmap,os
from pathlib import Path
import numpy as np
from datasets.m8j12_contract import CAMERAS
from tools.m8j13_common import atomic

SHAPE=(6,396,704,4)
SIZE=int(np.prod(SHAPE))

def binding(row,identity):
    return hashlib.sha256(json.dumps(dict(row=row,identity=identity),sort_keys=True,separators=(',',':')).encode()).hexdigest()

class Slots:
    def __init__(self,root,create=False):
        self.root=Path(root);self.leases={};self.generations=[0,0]
        if create:
            self.root.mkdir(mode=0o700)
            for i in range(2):
                fd=os.open(str(self.root/('%d.raw'%i)),os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
                os.ftruncate(fd,SIZE);os.close(fd)
    def publish(self,request,row,payloads,identity):
        slot=next((i for i in range(2) if i not in self.leases),None)
        if slot is None:return None
        if request in self.leases.values():raise RuntimeError('DuplicateRequest')
        self.generations[slot]+=1
        path=self.root/('%d.raw'%slot)
        with path.open('r+b') as f:
            with mmap.mmap(f.fileno(),SIZE,access=mmap.ACCESS_WRITE) as view:
                n=SIZE//6
                for i,name in enumerate(CAMERAS):
                    raw=payloads[name]
                    if len(raw)!=n:raise ValueError('TransportSizeMismatch')
                    view[i*n:(i+1)*n]=raw
        header=dict(slot=slot,generation=self.generations[slot],request=request,row=row,identity=identity,
            shape=list(SHAPE),dtype='uint8',layout='NHWC_BGRA',camera_order=list(CAMERAS),bytes=SIZE,binding=binding(row,identity))
        atomic(self.root/('%d.json'%slot),header);self.leases[slot]=request
        return {k:header[k] for k in ('slot','generation','request','binding')}
    def release(self,request):
        found=[i for i,x in self.leases.items() if x==request]
        if len(found)!=1:raise RuntimeError('InvalidLeaseRelease')
        del self.leases[found[0]]
    def read(self,token):
        slot=token['slot']
        if type(slot)!=int or slot not in (0,1):raise ValueError('InvalidSlot')
        header=json.loads((self.root/('%d.json'%slot)).read_text())
        if any(header[k]!=token[k] for k in ('slot','generation','request','binding')):raise RuntimeError('StaleLease')
        if header['shape']!=list(SHAPE) or header['dtype']!='uint8' or header['layout']!='NHWC_BGRA' or header['camera_order']!=list(CAMERAS) or header['bytes']!=SIZE:raise ValueError('TransportLayoutMismatch')
        if binding(header['row'],header['identity'])!=header['binding']:raise RuntimeError('MetadataBindingMismatch')
        with (self.root/('%d.raw'%slot)).open('rb') as f:
            if os.fstat(f.fileno()).st_size!=SIZE:raise ValueError('TransportSizeMismatch')
            with mmap.mmap(f.fileno(),SIZE,access=mmap.ACCESS_READ) as view:
                arr=np.ndarray(SHAPE,np.uint8,buffer=view)
                images={n:arr[i,:,:,:3].copy() for i,n in enumerate(CAMERAS)}
                del arr
        if json.loads((self.root/('%d.json'%slot)).read_text())!=header:raise RuntimeError('LeaseOverwrittenDuringRead')
        return header['row'],images,header['identity']
    def close(self):
        for p in self.root.iterdir():p.unlink()
        self.root.rmdir()
