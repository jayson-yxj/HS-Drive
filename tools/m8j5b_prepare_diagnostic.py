"""Create four RGB-only numeric packets, using the existing J3 reader."""
import hashlib
import json
import numpy as np

from tools.m8j3_validate_clip import RAW,read_annotation,export_packet
from tools.m8j5b_build_coverage import ROOT,save
from datasets.m8j4_petr_input import validate_arrays


def main():
    folder=ROOT/'rgb_only';folder.mkdir(exist_ok=False)
    protocol=json.loads(open('configs/m8j5b_protocol.json').read());entries=[]
    for ordinal in protocol['bounded_diagnostics']['frozen_forward_ordinals']:
        rgb=export_packet(read_annotation(RAW/'anno'/(ordinal+'.json.gz')),ordinal,folder)
        validate_arrays(rgb['images'],rgb['intrinsics'],rgb['cam_to_ego'])
        path=ROOT/'inference_input'/(ordinal+'.npz')
        if path.exists():raise FileExistsError(path)
        np.savez(path,images=rgb['images'],intrinsics=rgb['intrinsics'],cam_to_ego=rgb['cam_to_ego'])
        entries.append(dict(ordinal=ordinal,path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(ROOT/'inference_input/manifest.json',dict(entries=entries,keys=['images','intrinsics','cam_to_ego'],GT=False))


if __name__=='__main__':main()
