"""Native rendered semantic source proportions for direct positives, train/val only."""
import json
from pathlib import Path
import numpy as np
from datasets.m5b_pairs import camera_rays
from datasets.m5a_training import VisualEvidenceDataset
from geometry.ray_tracing import VoxelGrid
from simulator.recorder import atomic_json,file_hash


def main():
    out=Path('reports/m5d_background_supervision.json')
    if out.exists():raise FileExistsError(out)
    index='/root/gpufree-data/hs-drive-data/processed/m5c_index';grid=VoxelGrid();records=[]
    for split in ['train','val']:
        ds=VisualEvidenceDataset(index,split)
        # Empty,near,medium,far,pedestrian fixed configurations; all groups.
        for i,row in enumerate(ds.rows):
            if row['configuration'] not in [0,1,3,5,11,13]:continue
            d={k:v.numpy() for k,v in ds[i].targets.items()};s=d['surface_semantic'];safe=d['obstacle_valid']&d['surface_semantic_valid']&~d['ego_mask']&~d['evidence_conflict_mask']&~d['semantic_conflict_mask'];pos=safe&s[1]&~s[0]&~s[2];frame=json.loads(Path(row['raw_frame']).read_text());root=Path(row['raw_frame']).parent.parent;man=json.loads((root/'manifest.json').read_text());categories={};actor_cells=np.zeros(grid.shape,bool)
            for name in row['camera_names']:
                rays,data=camera_rays(root,man,frame,name);ids,inside=grid.locate(rays['endpoints']);u,v=rays['uv'].T;keep=rays['reliable']&inside&((u+2*v)%4!=1);candidate=np.zeros(len(ids),bool)
                for actor in frame['actors']:
                    if actor['is_ego']:continue
                    box=np.array(actor['corners_hs_ego_m']);candidate|=((rays['endpoints']>=box.min(0)-.05)&(rays['endpoints']<=box.max(0)+.05)).all(-1)
                valid=np.flatnonzero(keep);valid=valid[pos.ravel()[ids[valid]]];np.logical_or.at(actor_cells.ravel(),ids[valid],candidate[valid]);tags=data[1][v,u]
                for tag in np.unique(tags[valid]):
                    cat=categories.setdefault(str(int(tag)),dict(pixels=0,cells=np.zeros(grid.shape,bool)));idx=valid[tags[valid]==tag];cat['pixels']+=len(idx);cat['cells'].ravel()[ids[idx]]=True
            records.append(dict(group=row['group'],split=split,configuration=row['configuration'],direct_positive=int(pos.sum()),placed_actor_associated_cells=int((pos&actor_cells).sum()),background_without_placed_actor=int((pos&~actor_cells).sum()),semantic_tags={k:dict(pixels=v['pixels'],unique_positive_cells=int(v['cells'].sum())) for k,v in categories.items()}));print(row['sample_id'],flush=True)
    atomic_json(out,dict(kind='offline_M5C_train_val_native_rendered_semantic_audit',records=records,source_sha256=file_hash(__file__),semantic_tag_definition='CARLA0.9.15 capture semantic IDs:3Building,4Wall,5Fence,6Pole,9Vegetation,12Pedestrian,14Car,15Truck,21Dynamic; raw IDs preserved',limits='Count of unique coarse cells per category can overlap; candidate AABB gates only real reliable surface points. Remaining background may include native map vehicles. Fixed subset configurations, not whole dataset census. No test data.'))

if __name__=='__main__':main()
