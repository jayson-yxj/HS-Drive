"""Official COCO Mask R-CNN, RGB only, zero CARLA fine-tuning."""
import argparse,json,time
from pathlib import Path
import torch,numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2,MaskRCNN_ResNet50_FPN_V2_Weights
from tools.m5f_common import inputs,pixel_label
from tools.m5f_dataset import ROOT
from tools.m5g_evaluation import summarize
from simulator.recorder import atomic_json,file_hash
from training.runtime import prepare_cuda


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=ROOT);p.add_argument('--split',default='val');a=p.parse_args();dest=Path(f'reports/m5g_pretrained_{a.split}.json')
    if dest.exists():raise FileExistsError(dest)
    if a.split=='test':
        from tools.m5g_common import verify_freeze
        verify_freeze()
    prepare_cuda();torch.set_num_threads(2);w=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1;path='/root/gpufree-data/hs-drive-cache/weights/m5g/'+w.url.split('/')[-1];net=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,box_score_thresh=.05);net.load_state_dict(torch.load(path,weights_only=True));net.cuda().eval();rows=[r for r in json.loads(Path(a.root,'index.json').read_text())['records']if r['split']==a.split];labels=[];scores=[];saved=[];times=[];out=Path(f'runs/m5g_pretrained_{a.split}');out.mkdir(exist_ok=False);torch.cuda.reset_peak_memory_stats()
    for row in rows:
        x=inputs(row);prob=np.zeros((2,6,384,640),np.float32);t0=time.perf_counter()
        with torch.no_grad():
            for c in range(6):
                y=net([x['images'][0,c]])[0]
                for k,ids in enumerate([[2,3,4,6,7,8],[1]]):
                    m=torch.isin(y['labels'],torch.tensor(ids,device='cuda'))
                    if m.any():prob[k,c]=(y['masks'][m,0]*y['scores'][m,None,None]).max(0).values.cpu().numpy()
        torch.cuda.synchronize();times.append((time.perf_counter()-t0)*1000);q=np.minimum(prob*256,255).astype(np.uint8);file=out/(row['sample_id'].replace('/','_')+'.npz');np.savez_compressed(file,vehicle=q[0],pedestrian=q[1]);saved.append(dict(sample_id=row['sample_id'],path=str(file),sha256=file_hash(file)));labels.append(pixel_label(row));scores.append(q.astype(np.float32)/256);print(row['sample_id'],times[-1],flush=True)
    locked=json.loads(Path('reports/m5g_pretrained_val.json').read_text())if a.split=='test'else None;result={}
    for key,index,classes in [('vehicle',0,(1,)),('pedestrian',1,(2,)),('supported',None,(1,2))]:
        probs=[s.max(0)if index is None else s[index]for s in scores];th=locked['results'][key]['threshold']if locked else None;result[key]=summarize(rows,labels,probs,th,classes)
    atomic_json(dest,dict(kind='actual_RTX4090_official_COCO_pretrained_noCARLA_finetune',results=result,saved=saved,latency_ms=dict(samples=times,p50=float(np.median(times[1:])),p95=float(np.quantile(times[1:],.95)),max=float(max(times[1:]))),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,weight_sha256=file_hash(path),source_sha256=file_hash(__file__),limits='Native384x640 resized800x1333; FP32 sequential6camera. Coneunsupported ignore notnegative. COCOexternalreference notsinglevariable. Instanceconfidence timesmaskprobability, notcalibratedoccupancy. GPUotherjobsmayoverlap; isolatedtimingseparate.'))

if __name__=='__main__':main()
