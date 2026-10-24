"""Read-only contract runner; refuses to run before original GPU forward passes."""
import argparse
import json
from pathlib import Path
from tools.m8a_p1_common import ROOT,LOG,run,write,provenance,asset

def main():
    p=argparse.ArgumentParser();p.add_argument('--forward-run',required=True);p.add_argument('--run-id',required=True);args=p.parse_args()
    result=json.loads((ROOT/'artifacts'/args.forward_run/'forward_results.json').read_text())
    assert result['status']=='Passed' and result['input_shape']==[1,6,3,512,1408]
    packet=Path('reports/m7a_shadow_packets/Town01_m5o_train_00_00074004.json')
    data=json.loads(packet.read_text());out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False)
    # No checkpoint/GT deserialization in this contract-only process; no CARLA client.
    env=dict(M8A_ISOLATED='1',M8A_OUTPUT=str(out),M8A_SHADOW_PACKET=str(packet.resolve()),PETR_SOURCE=str(ROOT/'sources/PETR'),
             PYTHONPATH=str(ROOT/'sources/mmdetection3d'),LD_LIBRARY_PATH=str(ROOT/'env/lib/python3.8/site-packages/torch/lib')+':'+str(ROOT/'cuda-11.8/lib'),OPENBLAS_NUM_THREADS='2')
    for camera in data['cameras']:assert asset(camera['path'])['sha256']==camera['sha256']
    run('camera_'+args.run_id,['unshare','--net',str(ROOT/'env/bin/python'),'-s',str(Path('tools/m8a_p1_camera_check.py').resolve())],extra_env=env,timeout=120)
    write(LOG/('camera_'+args.run_id+'_artifacts.json'),dict(**provenance('ActualOfflineContractExecution'),packet=asset(packet),files=[asset(x) for x in out.iterdir()]))
if __name__=='__main__':main()
