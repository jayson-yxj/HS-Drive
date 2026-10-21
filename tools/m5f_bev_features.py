"""Export old BEV and unmasked foreground channels for matched small fusion study."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from tools.m5f_common import inputs,pixel_label,checkpoint
from tools.m5a_learning import encoder_load
from tools.m5e_common import load_checkpoint
from models.m5f_pixel import ForegroundModel,PixelHead
from models.m5f_lift import lift
from models.m5b_cached_encoder import cache_rgb,cached_forward
from models.m5c_obstacle_head import VisibleObstacleHead
from tools.m5f_factorization import reference
from tools.m5f_dataset import ROOT
from training.runtime import prepare_cuda
from simulator.recorder import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',default=ROOT);p.add_argument('--split',nargs='+',default=['train','val']);p.add_argument('--output',default='runs/m5f_bev_features');a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2);rows=[r for r in json.loads(Path(a.index,'index.json').read_text())['records'] if r['split'] in a.split];models=json.loads(Path('reports/m5f_image_foreground_training.json').read_text())['records'];records=[]
    for seed in [101,102]:
        record=next(r for r in models if r['name']=='P8_fpn' and r['seed']==seed);ck=checkpoint(record);pn=encoder_load();pn.encoder.lateral.load_state_dict(ck['lateral']);pn.encoder.output.load_state_dict(ck['output']);ph=PixelHead().cuda();ph.load_state_dict(ck['head']);pixel=ForegroundModel(pn.encoder,ph,8).eval();net=encoder_load();old=load_checkpoint(f'runs/m5e_learning/m5e_A_base_{91 if seed==101 else 92}.pt');net.context_head.load_state_dict(old['context']);net.depth_head.load_state_dict(old['depth']);head=VisibleObstacleHead().cuda();head.load_state_dict(old['head']);head.eval()
        for row in rows:
            inp=inputs(row);d=pixel_label(row);target,valid,actors=reference(d,inp,row,row['camera_names'])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                p=pixel(**inp).float().sigmoid();c=cache_rgb(net,inp);oldlogits=head(cached_forward(net,c))['visible_obstacle'][0].float();fg=p[:,:,8::16,8::16];e=lift(net.splat,fg,c['probs'],c['K'],c['T'],c['image_hw'])
            path=out/f"{seed}_{row['sample_id'].replace('/','_')}.npz";np.savez_compressed(path,old_logit=oldlogits.cpu().numpy(),foreground_mass=e['foreground_mass'][0].cpu().numpy(),total_mass=e['total_depth_mass'][0].cpu().numpy(),target=target,valid=valid)
            records.append(dict(sample_id=row['sample_id'],group=row['group'],split=row['split'],configuration=row['configuration'],seed=seed,path=str(path),sha256=file_hash(path)));print(seed,row['sample_id'],flush=True)
    atomic_json(out/'index.json',dict(records=records,source_sha256=file_hash(__file__),limits='OriginalA_base oldfeatures vsfused. Foreground P8decoder sampledcommonS16pixels; originalDepth S16 frozen. Newscalarfusion affects evidence only, no defaultBEVencoder change. GT only storedtargets.'))

if __name__=='__main__':main()
