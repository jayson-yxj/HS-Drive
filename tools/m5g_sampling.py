"""Dense versus phased sparse Oracle sampling: renderer self-consistency only."""
import json,argparse
from pathlib import Path
import numpy as np
from geometry.reference import transform_points,unproject_pixels_to_optical
from geometry.ray_tracing import VoxelGrid
from diagnostics.m5c_localization import matching
from tools.m5f_common import pixel_label,inputs
from tools.m5f_factorization import reference
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/root/gpufree-data/hs-drive-data/processed/m5f_test_pixels');p.add_argument('--report',default='reports/m5g_oracle_sampling_ceiling.json');a=p.parse_args();dest=Path(a.report)
    if dest.exists():raise FileExistsError(dest)
    rows=json.loads(Path(a.root,'index.json').read_text())['records'];grid=VoxelGrid();records=[]
    for row in rows:
        inp=inputs(row,'cpu');d=pixel_label(row);target,valid,actors=reference(d,inp,row,row['camera_names']);truth=target&valid;v,u=np.mgrid[:384,:640];ids=[];good=[]
        for c in range(6):
            pts=transform_points(unproject_pixels_to_optical(u,v,d['depth'][c],inp['intrinsics'][0,c].numpy(),'optical_z'),inp['cam_to_ego'][0,c].numpy());rank,inside=grid.locate(pts);rank=rank.reshape(384,640);inside=inside.reshape(384,640);ids.append(rank);good.append(inside&d['valid'][c]&d['target'][c])
        for stride in [1,4,8,16]:
            for phase in ([0]if stride==1 else [0,stride//2,stride-1]):
                pred=np.zeros(grid.shape,bool);cams=[]
                for c,name in enumerate(row['camera_names']):
                    mask=good[c][phase::stride,phase::stride];r=ids[c][phase::stride,phase::stride][mask];cp=np.zeros(grid.shape,bool);cp.reshape(-1)[r]=True;pred|=cp;cams.append(dict(camera=name,native_target_pixels=int(good[c].sum()),sampled_target_rays=int(mask.sum()),covered_gt_cells=int((cp&truth).sum())))
                hit=pred&valid;am=[]
                for actor in actors:
                    m=actor['mask'];am.append(dict(type_id=actor['type_id'],distance_m=actor['distance_m'],visible_cells=int(m.sum()),covered=int((hit&m).sum()),matching=matching(hit,m)))
                records.append(dict(group=row['group'],sample_id=row['sample_id'],configuration=row['configuration'],stride=stride,phase=phase,native_target_pixels=int(sum(x.sum()for x in good)),sampled_rays=sum(c['sampled_target_rays']for c in cams),gt_cells=int(truth.sum()),covered_gt_cells=int((hit&truth).sum()),uncovered_cells=int((truth&~hit).sum()),matching=matching(hit,truth),by_camera=cams,actors=am))
        print(row['sample_id'],flush=True)
    aggregate=[]
    for s,phase in sorted({(r['stride'],r['phase'])for r in records}):
        rr=[r for r in records if(r['stride'],r['phase'])==(s,phase)];n=sum(r['gt_cells']for r in rr);aggregate.append(dict(stride=s,phase=phase,gt_cells=n,covered=sum(r['covered_gt_cells']for r in rr),uncovered=sum(r['uncovered_cells']for r in rr),sampled_rays=sum(r['sampled_rays']for r in rr),exact_recall=sum(r['covered_gt_cells']for r in rr)/n,recall_05=sum(r['matching']['tolerances']['0.5']['matched_gt']for r in rr)/n,recall_1m=sum(r['matching']['tolerances']['1.0']['matched_gt']for r in rr)/n))
    assert next(r for r in aggregate if r['stride']==1)['exact_recall']==1.
    atomic_json(dest,dict(kind='offline_CARLA_recorded_Oracle_sampling_diagnostic',aggregate=aggregate,records=records,index_sha256=file_hash(Path(a.root,'index.json')),source_sha256=file_hash(__file__),limits='M5F openedtest historicalregression. DenseOracle sharesrendererandreference notindependenttruth. Stride4 samplingonly notneuralnetwork. Hitanyray vsreference>=3nativepoints. Samehalfopengrid+knownmask; mixedcells excluded. Phases diagonaloffset examples notall possiblephasepairs.'))

if __name__=='__main__':main()
