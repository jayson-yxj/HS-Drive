"""Trace the fixed official input path; synthetic serialization is not training PKL proof."""
import ast
import json
import pickle
import subprocess
import tempfile
from pathlib import Path
from tools.m8d_common import *

def main():
    official=RUNTIME/'sources/PETR'
    files=['tools/data_converter/nuscenes_converter.py','tools/data_converter/nuscenes_converter_seg.py',
           'tools/generate_sweep_pkl.py','projects/mmdet3d_plugin/datasets/nuscenes_dataset.py',
           'projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py',
           'projects/mmdet3d_plugin/datasets/pipelines/loading.py',
           'projects/mmdet3d_plugin/models/utils/positional_encoding.py',
           'projects/mmdet3d_plugin/models/dense_heads/petr_head.py',
           'projects/configs/petr/petr_r50dcn_gridmask_c5.py']
    assert subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()==CFG['petr_commit']
    assignments={}
    for filename in files[:3]:
        tree=ast.parse((official/filename).read_text())
        assignments[filename]=[dict(line=n.lineno,name=t.id,value=ast.literal_eval(n.value))
            for n in ast.walk(tree) if isinstance(n,ast.Assign)
            for t in n.targets if isinstance(t,ast.Name) and t.id in ['camera_types','sensors']]
    names=assignments[files[0]][0]['value']
    assert names==['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
    import mmcv
    import numpy as np
    import projects.mmdet3d_plugin
    from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
    info=dict(token='synthetic',lidar_path='unused',sweeps=[],timestamp=1_000_000,cams={})
    for i,name in enumerate(names):
        info['cams'][name]=dict(timestamp=1_000_000,data_path=name+'.png',
            sensor2lidar_rotation=np.eye(3),sensor2lidar_translation=np.array([float(i),0,0]),
            cam_intrinsic=np.diag([i+2.,i+3.,1]))
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        path=str(Path(directory)/'synthetic.pkl')
        mmcv.dump(dict(infos=[info]),path)
        loaded=mmcv.load(path)['infos'][0]
    dataset=object.__new__(CustomNuScenesDataset)
    dataset.data_infos=[loaded];dataset.modality={'use_camera':True};dataset.test_mode=True
    data=dataset.get_data_info(0)
    assert data['img_filename']==[n+'.png' for n in names]
    for i in range(6):
        assert data['intrinsics'][i][0,0]==i+2
        np.testing.assert_allclose(data['lidar2img'][i],data['intrinsics'][i]@data['extrinsics'][i].T)
    roots=[Path('/root/gpufree-data'),Path.cwd()]
    available=sorted(set(str(p) for root in roots for p in root.rglob('*nuscenes*infos*train*.pkl')))
    write('reports/m8d_source_evidence.json',dict(**provenance('OfficialSourceAuditAndSyntheticDatasetTest'),
        official_sources=[asset(official/f) for f in files],
        framework_sources=[asset(RUNTIME/'sources/mmdetection3d/mmdet3d/datasets/pipelines'/f)
                           for f in ['loading.py','formating.py']],
        assignments=assignments,synthetic_pickle_order_preserved=True,actual_dataset_get_data_info_order=names,
        training_PKL_candidates=available,training_order='TrainingPKLOrderNotVerified',
        B_name='OfficialConverterOrderCandidate',
        limit='Synthetic trusted local pickle proves mechanism only. No actual checkpoint-training PKL obtained. Sweep generator replaces sweeps, not keyframe cams; sweep loader explicitly selects sensors. C5 singleframe config has no sweep loader.'))
    print('Official chain and synthetic serialization/get_data_info passed')
if __name__=='__main__':main()
