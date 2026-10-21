"""Offline RGB-only teacher cache including original instance masks and scores."""
import argparse,json,time
from pathlib import Path
import torch,numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2,MaskRCNN_ResNet50_FPN_V2_Weights
from tools.m5i_data import rows,ROOT
from tools.m5f_common import inputs
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=ROOT);p.add_argument('--split',default='trainval');a=p.parse_args();
    if a.split=='test':
        from tools.m5i_freeze import verify
        verify()
    out=Path('runs/m5i_teacher_'+a.split);out.mkdir(exist_ok=False);prepare_cuda();torch.set_num_threads(2)
    w=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1;path=Path('/root/gpufree-data/hs-drive-cache/weights/m5g',w.url.split('/')[-1]);net=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,box_score_thresh=.05).cuda().eval();net.load_state_dict(torch.load(path,weights_only=True));records=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for row in rows(a.root):
        if a.split!='trainval' and row['split']!=a.split:continue
        if a.split=='trainval' and row['split']not in ['train','val']:raise ValueError('NoTestteacherbeforefreeze')
        x=inputs(row);payload={};prob=np.zeros((2,6,384,640),np.float32);trusted=np.zeros_like(prob)
        with torch.no_grad():
            for c in range(6):
                y=net([x['images'][0,c]])[0];support=torch.isin(y['labels'],torch.tensor([1,2,3,4,6,7,8],device='cuda'));y={k:v[support]for k,v in y.items()}
                for k in ['labels','scores','boxes']:payload[f'c{c}_{k}']=y[k].cpu().numpy()
                payload[f'c{c}_masks']=np.round(y['masks'][:,0].cpu().numpy()*255).astype(np.uint8)
                for k,ids in enumerate([[2,3,4,6,7,8],[1]]):
                    mask=torch.isin(y['labels'],torch.tensor(ids,device='cuda'));conf=mask&(y['scores']>=.7)
                    if mask.any():prob[k,c]=(y['masks'][mask,0]*y['scores'][mask,None,None]).max(0).values.cpu().numpy()
                    if conf.any():trusted[k,c]=(y['masks'][conf,0]*y['scores'][conf,None,None]).max(0).values.cpu().numpy()
        pathout=out/(row['sample_id'].replace('/','_')+'.npz');np.savez_compressed(pathout,**payload,score=np.minimum(prob*256,255).astype(np.uint8),trusted=np.minimum(trusted*256,255).astype(np.uint8));records.append(dict(sample_id=row['sample_id'],split=row['split'],path=str(pathout),sha256=file_hash(pathout),input_sha256=row['input_sha256'],label_sha256=row['pixel_sha256']));print(row['sample_id'],flush=True)
    import inspect,torchvision
    atomic_json(out/'index.json',dict(kind='actual_RTX4090_RGB_only_offline_teacher',records=records,weight_sha256=file_hash(path),weight_url=w.url,source_sha256=file_hash(__file__),torchvision_source_sha256=file_hash(inspect.getfile(maskrcnn_resnet50_fpn_v2)),torchvision_version=torchvision.__version__,native_hw=[384,640],resize=dict(min_size=800,max_size=1333,actual_hw=[799,1333],padded_hw=[800,1344],batch_padding_divisor=32),score='max(instance score * mask probability), quantizedfloor256; instance mask round255',elapsed_s=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,unsupported='cone ignored byteacher; retain GT',license_provenance='reports/m5g_pretrained_provenance.json'))
if __name__=='__main__':main()
