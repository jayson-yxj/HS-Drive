"""Actual file I/O probe with synthetic fixtures; no captured data or inference."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
from datasets.m8i1_contract import ORDER, RGBInferenceLoader
from tools.m8i1_common import write, provenance


def main():
    opened=[]
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            opened.append(os.fsdecode(args[0]))
    sys.addaudithook(audit)
    with tempfile.TemporaryDirectory(prefix='m8i1-isolation-') as directory:
        root=Path(directory);root.chmod(0o755)
        rgb=root/'RGBInferencePacket';rgb.mkdir(mode=0o755)
        side=root/'EvaluatorSidecar';side.mkdir(mode=0o700)
        secret=side/'actor_GT.json';secret.write_text('{"GTActorCenter":[12,3,1]}');secret.chmod(0o600)
        cameras=[]
        for i,name in enumerate(ORDER):
            path=rgb/(name+'.png');cv2.imwrite(str(path),np.full((384,640,3),i*17,np.uint8));path.chmod(0o644)
            cameras.append(dict(name=name,path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                K=[[250,0,319.5],[0,250,191.5],[0,0,1]],T_cam_to_ego=np.eye(4).tolist(),frame_id=17,timestamp_s=.85))
        packet=dict(schema='RGBInferencePacket.m8i1.v1',source='CARLA0.9.15_RGBOnly',calibration_sha256='a'*64,
                    camera_order=ORDER,image_size=[384,640],frame_id=17,timestamp_s=.85,cameras=cameras,image_transform='Identity')
        f=rgb/'packet.json';f.write_text(json.dumps(packet));loader=RGBInferenceLoader(rgb)
        opened.clear();images=loader.load('packet.json');normal=list(opened)
        rows=[]
        for mode in ['GTField','ParentTraversal','AbsoluteSidecar','SymlinkSidecar','NestedGT','WrongTimestamp']:
            p=json.loads(json.dumps(packet))
            if mode=='GTField':p['ActorGT']={'center':[1,2,3]}
            if mode=='ParentTraversal':p['cameras'][0]['path']='../EvaluatorSidecar/actor_GT.json'
            if mode=='AbsoluteSidecar':p['cameras'][0]['path']=str(secret)
            if mode=='SymlinkSidecar':
                (rgb/'escape.png').symlink_to(secret);p['cameras'][0]['path']='escape.png'
            if mode=='NestedGT':p['cameras'][0]['GT']=str(secret)
            if mode=='WrongTimestamp':p['cameras'][0]['timestamp_s']=.90
            f.write_text(json.dumps(p));opened.clear()
            try:loader.load('packet.json');rejected=False;reason=None
            except (ValueError,TypeError) as exc:rejected=True;reason=str(exc)
            reads=list(opened);gt_read=any('EvaluatorSidecar' in x or 'escape.png' in x for x in reads)
            rows.append(dict(case=mode,rejected=rejected,error=reason,GT_path_opened=gt_read,
                             opened=[Path(x).name for x in reads]))
            assert rejected and not gt_read
        # OS isolation is a separate, actual unprivileged probe, not inferred from schema.
        command=['runuser','-u','nobody','--','/usr/bin/python3','-c',
                 'import pathlib,sys; pathlib.Path(sys.argv[1]).read_bytes()',str(secret)]
        denied=subprocess.run(command,capture_output=True,text=True)
        allowed=subprocess.run(command[:-1]+[str(rgb/'front.png')],capture_output=True,text=True)
        assert denied.returncode!=0 and 'PermissionError' in denied.stderr and allowed.returncode==0
        write('reports/m8i1_rgb_input_isolation.json',dict(**provenance('ActualFileIO_SyntheticFixtures_NoCARLAImages'),
              status='SyntheticContractAndOSProbeGo',real_pilot_packet_status='NotRun_NoImages',
              normal_shape=list(images.shape),normal_opened=[Path(x).name for x in normal],
              forbidden_cases=rows,unprivileged_sidecar_probe=dict(returncode=denied.returncode,error='PermissionError'),
              unprivileged_RGB_probe=dict(returncode=allowed.returncode),
              fixture_origin='Generated solid-color arrays and synthetic GT; not captured RGB, not model forward',
              scope='Whitelist path enforcement and OS separation tested on fixtures. Root/unrestricted arbitrary code is not sandboxed. Actual captured packet probe NotRun.',
              fixture_cleanup=True))


if __name__=='__main__':main()
