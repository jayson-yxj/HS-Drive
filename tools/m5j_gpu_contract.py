"""Actual GPU dualbranch isolation and legacy-lift equivalence."""
import torch,json
from pathlib import Path
from tools.m5j_common import rows,write
from tools.m5f_common import inputs,pixel_label
from models.m5j_candidate import MobileBEVCandidate
from models.m5f_lift import lift
from simulator.recorder import file_hash
from training.runtime import prepare_cuda

def main():
    prepare_cuda();torch.set_num_threads(2);row=rows('val')[2];x=inputs(row);d=pixel_label(row);records=[]
    for mode in ['mobile_gt','legacy']:
        for seed in [101,102]:
            net=MobileBEVCandidate(enable=True,visual=mode,seed=seed);a=net(**x);d['target'][...]=False;d['depth'][...]=0;b=net(**x);assert torch.equal(a['pixel_foreground'],b['pixel_foreground']);assert torch.equal(a['depth_probability'],b['depth_probability']);assert torch.allclose(a['evidence']['scores']['raw'],b['evidence']['scores']['raw']);old=lift(net.depth.splat,a['pixel_foreground'][:,:,4::8,4::8],a['depth_probability'],x['intrinsics'],x['cam_to_ego'],(384,640));assert torch.allclose(a['evidence']['scores']['raw'],old['event_score'],atol=1e-6);assert a['evidence']['foreground_mass'].dtype==torch.float32
            rejected=[]
            for field in ['depth','instance','actor','pair_witness','surface_valid']:
                try:net(**dict(x,**{field:torch.zeros(1,device='cuda')}))
                except ValueError:rejected.append(field)
            assert len(rejected)==5;records.append(dict(visual=mode,seed=seed,gt_mutation_identical=True,raw_old_maxdiff=float((old['event_score']-a['evidence']['scores']['raw']).abs().max()),rejected=rejected,shape=list(old['event_score'].shape),geometry_dtype=str(a['evidence']['foreground_mass'].dtype)))
    write('reports/m5j_mobile_migration.json',dict(kind='actual_RTX4090_dualbranch_contract_execution',records=records,passed=True,enabled_by_default=False,production_driver_changed=False,source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
