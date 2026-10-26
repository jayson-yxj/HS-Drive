"""CPU interface gate then prepare an RGB-only numeric bundle, no model."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from datasets.m8j3_rgb_clip import load_rgb_packet
from datasets.m8j4_petr_input import validate_arrays


def main():
    os.umask(0o077)
    cfg=json.loads(Path('configs/m8j4_protocol.json').read_text());root=Path(cfg['private_root'])
    gate=json.loads((root/'reviewed_summary.json').read_text())
    assert gate['frames']==214 and gate['anomalies']==23 and gate['active_ambiguous_matches']==0
    assert gate['status']=='BasicSidecarInterfaceGo_QualityPartial' and gate['stats']['positive_candidates']>0
    out=root/'inference_input';out.mkdir(exist_ok=False)
    entries=[]
    for ordinal in cfg['model']['ordinals']:
        rgb=load_rgb_packet(Path(cfg['source_root'])/'rgb_only',ordinal+'.json')
        validate_arrays(rgb['images'],rgb['intrinsics'],rgb['cam_to_ego'])
        path=out/(ordinal+'.npz')
        np.savez(path,images=rgb['images'],intrinsics=rgb['intrinsics'],cam_to_ego=rgb['cam_to_ego'])
        entries.append(dict(ordinal=ordinal,path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            keys=['images','intrinsics','cam_to_ego'],camera_order=list(rgb['camera_order']),
            source_frame=None,timestamp_s=None))
    with (out/'manifest.json').open('x') as f:
        json.dump(dict(entries=entries,model=cfg['model'],label_gate='BasicGeometryAndIsolationPassed_QualityPartial',
                       inference_input_no_GT=True),f,indent=2)
    print(json.dumps(dict(status='SingleFrameInputGatePassed',entries=3,GT_in_input=False)))


if __name__=='__main__':main()
