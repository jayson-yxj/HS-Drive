import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import cv2
from datasets.schema import SensorSample
from simulator.synchronizer import FrameSynchronizer
from datasets.m8i1_contract import REQUIRED, ORDER, RGBInferenceLoader, schedule_frames


def sample(name,frame,time=None):
    return SensorSample(name,frame,frame*.05 if time is None else time,1,1,b'\0\0\0\xff',np.eye(4))


class ContractTests(unittest.TestCase):
    def test_schedule_odd_and_even_phase_no_round_to_even(self):
        offsets=list(range(12))
        self.assertEqual(schedule_frames(101,180,offsets)[0],181)
        self.assertEqual(schedule_frames(100,180,offsets)[0],182)

    def test_24_channel_actual_phase_out_of_order(self):
        sync=FrameSynchronizer(REQUIRED)
        for f in range(101,104):sync.add_snapshot(f,f*.05,{},[])
        for n in reversed(REQUIRED):sync.callback(n)(sample(n,101))
        self.assertEqual([b.frame_id for b in sync.drain()],[101])
        for n in REQUIRED:sync.callback(n)(sample(n,103))
        self.assertEqual([b.frame_id for b in sync.drain()],[103])

    def test_one_missing_not_replaced_by_next_frame(self):
        events=[]; sync=FrameSynchronizer(REQUIRED,timeout_ticks=3,on_event=events.append)
        for f in range(101,105):
            sync.add_snapshot(f,f*.05,{},[])
            if f==101:
                for n in REQUIRED[:-1]:sync.callback(n)(sample(n,f))
            if f==102:sync.callback(REQUIRED[-1])(sample(REQUIRED[-1],f))
            self.assertEqual(sync.drain(),[])
        self.assertTrue(any(x['reason']=='dropped_frame' and x['frame_id']==101 for x in events))

    def test_mixed_timestamp_and_duplicate_reject(self):
        sync=FrameSynchronizer(REQUIRED)
        sync.add_snapshot(1,.05,{},[])
        for n in REQUIRED:sync.callback(n)(sample(n,1,.1 if n==REQUIRED[-1] else .05))
        sync.callback(REQUIRED[0])(sample(REQUIRED[0],1))
        self.assertFalse(sync.drain());self.assertEqual(sync.stats['invalid_frame'],1)
        self.assertEqual(sync.stats['duplicate_sensor'],1)

    def test_tick_gap(self):
        sync=FrameSynchronizer(REQUIRED);sync.add_snapshot(1,.05,{},[])
        with self.assertRaises(RuntimeError):sync.add_snapshot(3,.15,{},[])

    def fixture(self,root):
        cams=[]
        for n in ORDER:
            p=root/(n+'.png');cv2.imwrite(str(p),np.zeros((384,640,3),np.uint8))
            cams.append(dict(name=n,path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                        K=np.eye(3).tolist(),T_cam_to_ego=np.eye(4).tolist(),frame_id=1,timestamp_s=.05))
        return dict(schema='RGBInferencePacket.m8i1.v1',source='CARLA0.9.15_RGBOnly',
                    calibration_sha256='1'*64,camera_order=ORDER,image_size=[384,640],frame_id=1,
                    timestamp_s=.05,cameras=cams,image_transform='Identity')

    def test_actual_file_whitelist_and_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'rgb';root.mkdir();side=Path(d)/'evaluator';side.mkdir()
            (side/'secret.json').write_text('{"GT":42}')
            p=self.fixture(root);file=root/'packet.json';file.write_text(json.dumps(p));loader=RGBInferenceLoader(root)
            self.assertEqual(loader.load(file.name).shape,(6,384,640,3))
            p['GT']={'actor':42};file.write_text(json.dumps(p))
            with self.assertRaises(ValueError):loader.load(file.name)
            del p['GT'];p['cameras'][0]['path']='../evaluator/secret.json';file.write_text(json.dumps(p))
            with self.assertRaises(ValueError):loader.load(file.name)
            (root/'escape.png').symlink_to(side/'secret.json');p['cameras'][0]['path']='escape.png';file.write_text(json.dumps(p))
            with self.assertRaises(ValueError):loader.load(file.name)

    def test_only_owner_ticks_no_legacy_placer_or_model(self):
        forbidden={'cast_ray','set_simulate_physics','apply_physics_control','set_transform',
                   'enable_constant_velocity','load_world','reload_world','apply_batch_sync'}
        for p in [*Path('simulator').glob('m8i1_*.py'),*Path('datasets').glob('m8i1_*.py')]:
            t=ast.parse(p.read_text())
            calls={n.func.attr for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
            self.assertFalse(calls&forbidden,(p,calls&forbidden))
            self.assertNotIn('import torch',p.read_text())

    def test_protocol_limits_fixed_before_run(self):
        p=json.loads(Path('configs/m8i1/protocol.json').read_text())
        self.assertEqual((p['width'],p['height'],p['sensor_tick'],p['fixed_delta_seconds']),(640,384,.1,.05))
        self.assertEqual((p['vehicle_attempts'],p['walker_attempts']),(8,12))
        self.assertLessEqual(p['init_ticks']+p['observation_ticks'],700)
        self.assertEqual(len(p['save_offsets_sensor_periods']),12)


if __name__=='__main__':unittest.main()
