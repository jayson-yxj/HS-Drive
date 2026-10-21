"""Offline foreground-selected S4 ray feasibility, oracle depth diagnostic only."""
import json,time
from pathlib import Path
import numpy as np
from geometry.reference import transform_points,unproject_pixels_to_optical
from geometry.ray_tracing import VoxelGrid
from tools.m5f_common import inputs,pixel_label
from tools.m5f_factorization import reference
from diagnostics.m5c_localization import matching
from simulator.recorder import atomic_json,file_hash
from tools.m5h_freeze import verify
from tools.m5h_shadow import TEST

def main():
    dest=Path('reports/m5h_roi_diagnostics.json')
    if dest.exists():raise FileExistsError(dest)
    frozen=verify();rows=json.loads(Path(TEST,'index.json').read_text())['records'];pix=json.loads(Path('reports/m5h_pixel_test.json').read_text())['records'];grid=VoxelGrid();records=[]
    for seed in [101,102]:
        choice=frozen['choices'][str(seed)];model=next(r for r in pix if r['name']==choice['pixel']and r['seed']==seed);lookup={r['sample_id']:r for r in model['saved']};threshold=int(choice['pixel_threshold']*256)/256
        for row in rows:
            d=pixel_label(row);inp=inputs(row,'cpu');truth,known,actors=reference(d,inp,row,row['camera_names']);p=np.load(lookup[row['sample_id']]['path'])['probability_u8'].astype(np.float32)/256
            for mode,stride in [('full_P8',8),('full_S4',4),('student_ROI_S4',4)]:
                t0=time.perf_counter();pred=np.zeros(grid.shape,bool);total=targetn=targethit=0;v,u=np.mgrid[stride//2:384:stride,stride//2:640:stride]
                for c in range(6):
                    roi=(p[c,v,u]>=threshold)if mode.startswith('student')else np.ones_like(u,bool);total+=int(roi.sum());gt=d['valid'][c,v,u]&d['target'][c,v,u];targetn+=int(gt.sum());targethit+=int((gt&roi).sum())
                    # ROI determined BEFORE any supervision. Depth is only an oracle intervention.
                    pts=transform_points(unproject_pixels_to_optical(u[roi],v[roi],d['depth'][c,v[roi],u[roi]],inp['intrinsics'][0,c].numpy(),'optical_z'),inp['cam_to_ego'][0,c].numpy());ids,inside=grid.locate(pts);valid=d['valid'][c,v[roi],u[roi]];fg=gt[roi]if not mode.startswith('student')else np.ones(int(roi.sum()),bool);pred.reshape(-1)[ids[inside&valid&fg]]=True
                records.append(dict(seed=seed,group=row['group'],sample_id=row['sample_id'],configuration=row['configuration'],mode=mode,ray_count=total,s4_target_rays=targetn,roi_target_rays=targethit,roi_missed_fraction=1-targethit/targetn if targetn else None,matching=matching(pred&known,truth&known),known_background_cells=int((pred&known&~truth).sum()),cpu_geometry_ms=(time.perf_counter()-t0)*1000,threshold=threshold))
    atomic_json(dest,dict(kind='offline_CPU_recorded_RGB_ROI_and_OracleDepth_sampling',records=records,source_sha256=file_hash(__file__),limits='ROIs use only frozen student output; GT usedonlydiagnosticcoverage/ref. fullS4/P8 are Oracleforeground ceilings, notlearned baselines. CPU geometry cost excludesneural denseDepth/highresolutionfeaturecost. No S4 learnedDepth/noSplatdefaultchange. GTbaddepthmask afterROI onlyforOracleprojection.'))
if __name__=='__main__':main()
