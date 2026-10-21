"""Matched RGB perturbation across existing Context/HeightSplat/BEV; descriptive norms."""
import json
from pathlib import Path
import numpy as np,torch
from tools.m5f_dataset import ROOT
from tools.m5f_common import inputs,pixel_label
from tools.m5a_learning import encoder_load
from tools.m5e_common import load_checkpoint
from models.m5b_cached_encoder import cache_rgb,cached_forward
from models.m5c_obstacle_head import VisibleObstacleHead
from tools.m5f_factorization import reference
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash


def main():
    out=Path('reports/m5f_feature_retention.json')
    if out.exists():raise FileExistsError(out)
    prepare_cuda();torch.set_num_threads(2);rows=[r for r in json.loads(Path(ROOT,'index.json').read_text())['records'] if r['split']=='val'];net=encoder_load();ck=load_checkpoint('runs/m5e_learning/m5e_A_base_91.pt');net.context_head.load_state_dict(ck['context']);head=VisibleObstacleHead().cuda();head.load_state_dict(ck['head']);records=[]
    for group in sorted({r['group'] for r in rows}):
        rr=sorted([r for r in rows if r['group']==group],key=lambda r:r['configuration']);inp=inputs(rr[0]);ca=cache_rgb(net,inp)
        with torch.no_grad():a=cached_forward(net,ca,return_parts=True)
        for row in rr[1:7]:
            inp=inputs(row);d=pixel_label(row);cb=cache_rgb(net,inp);target,valid,actors=reference(d,inp,row,row['camera_names']);m=d['target'][:,8::16,8::16]&d['valid'][:,8::16,8::16];am=np.any([v['mask'] for v in actors],axis=0) if actors else np.zeros_like(target)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):b=cached_forward(net,cb,return_parts=True);pa=head(a['bev'])['visible_obstacle'][0].float().sigmoid();pb=head(b['bev'])['visible_obstacle'][0].float().sigmoid()
            stages={}
            for name,av,bv,mask in [('image',ca['features'][0].float().norm(dim=1), (cb['features'][0].float()-ca['features'][0].float()).norm(dim=1),m),('context',a['context'][0].float().norm(dim=1),(b['context'][0].float()-a['context'][0].float()).norm(dim=1),m),('height',a['height'][0].float().norm(dim=0),(b['height'][0].float()-a['height'][0].float()).norm(dim=0),am),('bev',a['bev'][0].float().norm(dim=0),(b['bev'][0].float()-a['bev'][0].float()).norm(dim=0),am.any(0))]:
                mask=torch.from_numpy(mask).cuda();stages[name]=dict(support=int(mask.sum()),changed_region_delta_norm=float(bv[mask].mean()) if mask.any() else None,relative_delta=float((bv/(av+1e-5))[mask].mean()) if mask.any() else None,elsewhere_delta_norm=float(bv[~mask].mean()))
            records.append(dict(group=group,configuration=row['configuration'],stages=stages,actor_cells=int(am.sum()),oldhead_mean_delta=float((pb-pa)[torch.from_numpy(am).cuda()].mean()) if am.any() else None))
    atomic_json(out,dict(kind='actual_GPU_fixed_existing_oldBEV_path_val_empty_present_intervention',records=records,source_sha256=file_hash(__file__),limits='Stage norms differentunits, do not infer causal percent attenuation from normratio. BEV collapsesheight; mask knownactoronly. Originalmodel noOracledepth input. SharedRGB perturbation shows propagation not sufficient geometric correctness.'))

if __name__=='__main__':main()
