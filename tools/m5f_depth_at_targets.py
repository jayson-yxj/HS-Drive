"""Fixed original depth probabilities at reliable current target pixels, diagnostic only."""
import json
from pathlib import Path
import numpy as np,torch
from tools.m5f_common import inputs,pixel_label
from tools.m5a_learning import encoder_load
from models.m5b_cached_encoder import cache_rgb
from models.m5d_stride8 import cache_p8
from models.m5e_paths import geometry
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash


def main():
    dest=Path('reports/m5f_depth_at_targets.json')
    if dest.exists():raise FileExistsError(dest)
    prepare_cuda();torch.set_num_threads(2);net=encoder_load();rows=json.loads(Path('/root/gpufree-data/hs-drive-data/processed/m5f_test_pixels/index.json').read_text())['records'];records=[]
    for row in rows:
        inp=inputs(row);d=pixel_label(row)
        for s in [16,8]:
            c=(cache_rgb if s==16 else cache_p8)(net,inp);dep=torch.from_numpy(d['depth'][:,s//2::s,s//2::s][None]).cuda();g=geometry(net,c,dep);valid=torch.from_numpy((d['target']&d['valid'])[:,s//2::s,s//2::s][None]).cuda()&g['inside'];prob=c['probs'];centers=net.splat.depth_centers;exp=(prob*centers[None,None,:,None,None]).sum(2);mode=centers[prob.argmax(2)];values={'same_cell_mass':(prob*(g['ranks']==g['ids'][:,:,None])*g['bin_inside']).sum(2),'same_layer_1m_mass':(prob*g['same_layer']*(g['xy']<=1)).sum(2),'depth_pm1m_mass':(prob*((centers[None,None,:,None,None]-dep[:,:,None]).abs()<=1)).sum(2),'expected_depth_abs_error':(exp-dep).abs(),'mode_depth_abs_error':(mode-dep).abs(),'oracle_bin_xy_floor':g['xy'].masked_fill(~g['same_layer'],float('inf')).min(2).values}
            for lo,hi in [(0,5),(5,10),(10,15),(15,20),(20,35),(35,80)]:
                m=valid&(g['rad']>=lo)&(g['rad']<hi);n=int(m.sum());record=dict(group=row['group'],configuration=row['configuration'],stride=s,distance=f'{lo}_{hi}',n=n)
                for key,x in values.items():
                    xx=x[m];finite=torch.isfinite(xx);record[key]=float(xx[finite].mean()) if finite.any() else None;record[key+'_unsupported']=int((~finite).sum())
                records.append(record)
    atomic_json(dest,dict(kind='actual_GPU_frozen_depth_probability_at_rendered_visible_target_rays',records=records,source_sha256=file_hash(__file__),limits='Targetlabelonly selectsdiagnosticrays, neverforwardinput. S8evaluates originalfrozenDepthhead onnativeP8distribution; notnewlytrainedP8Depth. Expectations opticalZ metres. Bin floor meansgeometrydiscretization oracle, notlearned error. No newthresholdselection.'))

if __name__=='__main__':main()
