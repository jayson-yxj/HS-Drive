"""Frozen inference entrypoint. Takes only sanitized packet/model manifest."""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from PIL import Image
from datasets.m7a_shadow import validate_packet,digest
from datasets.m7a_contracts import Stamp,EGO,DetectedObjectHypothesis,RGBSpatialEvidence
from models.m5i_visual import MobilePixel
from models.m5n_direct import DirectCenterHead,feature_vector,reconstruct
from diagnostics.m5l_objects import components


@torch.no_grad()
def infer(packet,mobile,head,model_sha):
    validate_packet(packet,packet['timestamp_s'])
    images=np.stack([np.array(Image.open(c['path'])).transpose(2,0,1) for c in packet['cameras']])
    x=torch.from_numpy(images).cuda().float()[None]/255
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        feature=mobile.fuse(mobile.levels(x))
        probability=mobile.head(feature,(384,640)).float().sigmoid()
    q=(torch.clamp((probability*256).floor(),max=255)/256).reshape(6,384,640).cpu().numpy()
    objects=[]
    for ci,name in enumerate(packet['camera_order']):
        K=np.array(packet['intrinsics'][ci]);T=np.array(packet['cam_to_ego'][ci])
        for idx,c in enumerate(components(q[ci])):
            vector=feature_vector(feature[ci:ci+1],c,q[ci],K,T)
            point=reconstruct(head(vector[None]),torch.tensor([c['bbox']],device='cuda'),
              torch.tensor(K[None],device='cuda'),torch.tensor(T[None],device='cuda'))[0].cpu().tolist()
            objects.append(DetectedObjectHypothesis(name,packet['sample_id']+':101:'+name+':'+str(idx),tuple(c['bbox']),tuple(point)))
    torch.cuda.synchronize()
    elapsed=(time.perf_counter()-start)*1000
    out=RGBSpatialEvidence(Stamp(packet['frame_id'],packet['timestamp_s'],packet['valid_until_s'],EGO),
      packet['sample_id'],tuple(objects),model_sha)
    return dict(**asdict(out),source=out.source,safety_free=False,
       model_input_fields=['RGB','intrinsics','cam_to_ego'],inference_ms=elapsed,
       available_sim_timestamp='NotAvailable_offline_no_running_sim_clock',uncertainty_guarantee=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    manifest=json.loads(Path(a.manifest).read_text())
    if set(manifest)!={'packets','mobile','direct','qualification'}:raise ValueError('inference_manifest_fields')
    torch.set_num_threads(2);torch.manual_seed(101)
    from training.runtime import prepare_cuda
    prepare_cuda()
    for name in ['mobile','direct']:
        rec=manifest[name]
        if digest(rec['checkpoint'])!=rec['sha256']:raise ValueError('checkpoint_SHA')
    mobile=MobilePixel().cuda().eval();mobile.load_state_dict(torch.load(manifest['mobile']['checkpoint'],weights_only=True)['model'])
    head=DirectCenterHead(600).cuda().eval();head.load_state_dict(torch.load(manifest['direct']['checkpoint'],weights_only=True)['model'])
    torch.cuda.reset_peak_memory_stats();records=[]
    for rec in manifest['packets']:
        if digest(rec['path'])!=rec['sha256']:raise ValueError('packet_SHA')
        p=json.loads(Path(rec['path']).read_text());records.append(infer(p,mobile,head,manifest['direct']['sha256']))
    out=dict(kind='ActualRTX4090_FrozenRGBOfflineInference',source='RGB_estimate_OfflineShadow',qualification='CandidateNoGo',
      records=records,models=manifest,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
      device=torch.cuda.get_device_name(),training_steps=0,
      GT_read=False,ego_pose_not_consumed_by_model=True,control_authorized=False,source_sha256=digest(__file__))
    with Path(a.output).open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print('Frozen RGB forwards',len(records),'objects',sum(len(r['hypotheses']) for r in records),'peakMiB',out['peak_allocated_mib'])

if __name__=='__main__':main()
