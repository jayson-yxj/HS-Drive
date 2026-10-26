"""Append-only correction of coverage K/T to exact RGB preprocessing contract.

Retain first masks and failed assertion. Pixel Ignore arrays are unchanged.
"""
import hashlib
from pathlib import Path
import numpy as np

from datasets.m8j3_rgb_clip import CAMERAS
from datasets.m8j4_petr_input import A
from geometry.reference import CARLA_EGO_TO_HS_EGO as S,OPTICAL_TO_CARLA_CAM as O
from tools.m8j3_validate_clip import RAW,read_annotation
from tools.m8j5b_build_coverage import ROOT,save


def main():
    out=ROOT/'masks_v2';out.mkdir(exist_ok=False);maxdiff=0.;records=[]
    for old in sorted((ROOT/'masks').glob('*.npz')):
        a=read_annotation(RAW/'anno'/(old.stem+'.json.gz'));mat=[]
        for cam in CAMERAS:
            s=a['sensors'][cam];mat.append(A@np.asarray(s['intrinsic'])@np.linalg.inv(S@np.asarray(s['cam2ego'])@O)[:3])
        mat=np.stack(mat).astype(np.float32)
        with np.load(old,allow_pickle=False) as z:
            maxdiff=max(maxdiff,float(abs(z['image_projection']-mat).max()))
            np.savez_compressed(out/old.name,ignore=z['ignore'],image_projection=mat)
        records.append(dict(ordinal=old.stem,sha256=hashlib.sha256((out/old.name).read_bytes()).hexdigest()))
    save(ROOT/'mask_v2_manifest.json',records)
    save('reports/m8j5b_calibration_correction.json',dict(frames=len(records),max_projection_matrix_change=maxdiff,
        reason='Use exact cam2ego chain of RGB loader; raw world matrix cancellation magnifies source rounding',
        pixel_masks_changed=False,original_matrices_preserved=True,threshold_relaxed=False,
        new_contract='A @ K @ inverse(S @ cam2ego @ OpticalToCarla)[:3]',
        note='Matrix element difference is not pixel reprojection error'))


if __name__=='__main__':main()
