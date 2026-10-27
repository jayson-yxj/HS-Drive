"""Bounded FP32 eight-frame experiment. Original PETR inference and loss only."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch

from tools import m8j6_gpu_actual as old
from tools.m8j6_retry_gpu import training_mode, optimizer_step, evaluation_probe, tensor_output_compare
from training.m8j6_retry_state import (parameter_mapping, head_policy, snapshot, fingerprint,
    assert_exact, rng_state, restore_rng, optimizer_summary, assert_frozen_unchanged)
from training.m8j7_metrics import summarize, average_precision, prediction_mask
from datasets.m8j6_selection import sha256


def draw_corners(image, corners, projection, color):
    """Actual LiDARInstance3DBoxes corner order; near-plane and image clipping."""
    edges=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
    for corner in corners:
        q=np.c_[corner,np.ones(8)]@projection.T
        for i,j in edges:
            a,b=q[i,:3].copy(),q[j,:3].copy()
            if a[2]<.1 and b[2]<.1:continue
            if a[2]<.1:a+=(b-a)*((.1-a[2])/(b[2]-a[2]))
            if b[2]<.1:b+=(a-b)*((.1-b[2])/(a[2]-b[2]))
            u=tuple(np.clip(np.round(a[:2]/a[2]),-1000000,1000000).astype(int))
            v=tuple(np.clip(np.round(b[:2]/b[2]),-1000000,1000000).astype(int))
            ok,u,v=cv2.clipLine((0,0,image.shape[1],image.shape[0]),u,v)
            if ok:cv2.line(image,u,v,color,2,cv2.LINE_AA)


def visualize(directory,index,img,meta,gt,decoded,scores,threshold,cameras,step):
    images=img[0].detach().cpu().numpy().transpose(0,2,3,1)+np.array([103.530,116.280,123.675])
    gc=gt.corners.detach().cpu().numpy()
    boxes=decoded.tensor.detach().cpu().numpy()
    selected=prediction_mask(boxes,scores,threshold)
    pc=decoded.corners.detach().cpu().numpy()[selected] if len(decoded) else np.zeros((0,8,3))
    panels=[]
    for ci,name in enumerate(cameras):
        canvas=np.ascontiguousarray(np.clip(images[ci],0,255).astype(np.uint8))
        draw_corners(canvas,gc,meta[0]['lidar2img'][ci],(0,220,0))
        draw_corners(canvas,pc,meta[0]['lidar2img'][ci],(0,150,255))
        cv2.putText(canvas,f'{name} step{step} GT green / decoded orange >= {threshold}',
            (12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
        cv2.imwrite(str(directory/f'{index:02}_{name}.jpg'),canvas)
        panels.append(cv2.resize(canvas,(704,256)))
    cv2.imwrite(str(directory/f'{index:02}_six.jpg'),np.concatenate([np.concatenate(panels[:3],1),np.concatenate(panels[3:],1)],0))
    bev=np.full((1000,1000,3),245,np.uint8)
    for c,color in [(gc,(0,150,0)),(pc,(0,100,240))]:
        for box in c:
            xy=box[[0,3,7,4],:2]
            pixel=np.stack([500-xy[:,1]*8,500-xy[:,0]*8],axis=1).round().astype(np.int32)
            cv2.polylines(bev,[pixel],True,color,1,cv2.LINE_AA)
    cv2.circle(bev,(500,500),5,(0,0,0),-1)
    cv2.putText(bev,f'frame{index} step{step} x up / y left; GT green; decoded orange',
        (15,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
    cv2.imwrite(str(directory/f'{index:02}_bev.jpg'),bev)


def evaluate(model,prepared,manifest,protocol,private,report,step):
    state=rng_state();modes={n:m.training for n,m in model.named_modules()}
    before=fingerprint(model.state_dict());model.eval()
    directory=private/f'eval_{step:03}';directory.mkdir(exist_ok=False)
    records=[];public=[];start=time.perf_counter()
    try:
        with torch.no_grad():
            for frame in manifest['frames']:
                img,meta,gt,labels=old.load_frame(prepared,frame)
                raw=old.forward(model,img,meta)
                loss=model.pts_bbox_head.loss([gt],[labels],raw)
                for key,value in loss.items():old.finite(value,'eval_loss.'+key)
                layers=old.layer_stats(model.pts_bbox_head,raw,gt,labels)
                # Normal inference: no Hungarian target assignment used in selection.
                decoded,scores,pred_labels=model.pts_bbox_head.get_bboxes(raw,meta)[0]
                old.finite(decoded.tensor,'decoded_boxes');old.finite(scores,'decoded_scores')
                assert len(scores)<=protocol['evaluation']['max_num']
                assert (decoded.tensor[:,3:6]>0).all()
                r=dict(boxes=decoded.tensor.detach().cpu().numpy(),scores=scores.cpu().numpy(),
                    labels=pred_labels.cpu().numpy(),gt=gt.tensor.cpu().numpy(),gt_labels=labels.cpu().numpy())
                assert len(labels)==frame['retained_gt']
                records.append(r)
                np.savez_compressed(directory/f'{frame["index"]:02}_decoded_and_GT.npz',**r)
                visualize(directory,frame['index'],img,meta,gt,decoded,r['scores'],
                    protocol['evaluation']['score_threshold'],manifest['six_camera_order'],step)
                primary=summarize([r],manifest['training_classes'],.2,2.)
                public.append(dict(index=frame['index'],sample_token=frame['sample_token'],
                    losses={k:float(v) for k,v in loss.items()},total_loss=float(sum(loss.values())),
                    classification_loss=sum(float(v) for k,v in loss.items() if 'cls' in k),
                    regression_loss=sum(float(v) for k,v in loss.items() if 'bbox' in k),
                    layers=[{k:v for k,v in l.items() if k not in ('positive_indices','negative_indices')} for l in layers],
                    metrics=primary))
                print('eval',step,frame['index'],'TP',primary['TP'],'FP',primary['FP'],'FN',primary['FN'],flush=True)
        classes=manifest['training_classes'];spec=protocol['evaluation']
        result=dict(step=step,scope=spec['scope'],frames=public,
            mean_losses={k:float(np.mean([f['losses'][k] for f in public])) for k in public[0]['losses']},
            mean_total_loss=float(np.mean([f['total_loss'] for f in public])),
            mean_classification_loss=float(np.mean([f['classification_loss'] for f in public])),
            mean_regression_loss=float(np.mean([f['regression_loss'] for f in public])),
            primary=summarize(records,classes,spec['score_threshold'],2.),
            stricter={str(d):summarize(records,classes,spec['score_threshold'],d) for d in (1.,.5)},
            PR=[summarize(records,classes,t,2.) for t in spec['PR_score_thresholds']],
            AP=average_precision(records,classes),elapsed_s=time.perf_counter()-start,
            private_visualization_directory=str(directory),model_state_unchanged=before==fingerprint(model.state_dict()))
        assert result['primary']['GT']==335 and result['model_state_unchanged']
        html='<meta charset="utf-8"><h1>Private eight-frame training diagnostic</h1><p>Green GT; orange normal decoded predictions. No human signoff.</p>'
        for f in public:
            i=f['index'];html+=f'<h2>Frame {i} / step {step}</h2><img width="1500" src="{i:02}_six.jpg"><img width="700" src="{i:02}_bev.jpg">'
        (directory/'index.html').write_text(html)
        old.save(report/f'eval_{step:03}.json',result)
        return result
    finally:
        for n,m in model.named_modules():m.train(modes[n])
        restore_rng(state)
        assert_exact(state,rng_state(),'EvalPreservesRNG')
        assert {n:m.training for n,m in model.named_modules()}==modes


def save_checkpoint(model,optimizer,prepared,frames,mapping,original,step,private,report,protocol,commit):
    outputs=evaluation_probe(model,prepared,frames)
    payload=dict(model=snapshot(model.state_dict()),optimizer=snapshot(optimizer.state_dict()),
        global_step=step,next_sample_position=step%8,sample_order=list(range(8)),scheduler=None,
        rng=rng_state(),module_modes={n:m.training for n,m in model.named_modules()},mapping=mapping,
        original_state_sha256=old.state_sha(original),run_id=protocol['run_id'],code_commit=commit,
        protocol_sha256=sha256('configs/m8j7_protocol.json'),fixed_eval_outputs=outputs)
    path=private/f'checkpoint_{step:03}.pth'
    if path.exists():raise FileExistsError(path)
    torch.save(payload,path)
    result=dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path),step=step,
        model_fingerprint=fingerprint(payload['model']),optimizer_fingerprint=fingerprint(payload['optimizer']),
        mapping_fingerprint=fingerprint(mapping),rng_fingerprint=fingerprint(payload['rng']))
    old.save(report/f'checkpoint_{step:03}.json',result)
    return result


def main():
    protocol=json.loads(Path('configs/m8j7_protocol.json').read_text())
    cfg=json.loads(Path(protocol['original_protocol']).read_text())
    rt=Path(os.environ['M8J6_RUNTIME']);private=Path(os.environ['M8J7_PRIVATE']);report=Path(os.environ['M8J7_REPORT'])
    prepared=Path('/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/prepared')
    manifest=json.loads((prepared/'manifest.json').read_text());frames=manifest['frames']
    phase=os.environ['M8J7_PHASE'];commit=os.environ['M8J7_COMMIT']
    result=dict(phase=phase,run_id=protocol['run_id'],code_commit=commit,process_pid=os.getpid(),
        started_at_utc=datetime.now(timezone.utc).isoformat(),status='Running',optimizer_steps=0,
        precision='FP32',AMP=False,GradScaler='NotUsed',TF32=False,
        torch=torch.__version__,cuda=torch.version.cuda,mmcv=old.mmcv.__version__,mmdet=old.mmdet.__version__,
        mmdet3d=old.mmdet3d.__version__,GPU=torch.cuda.get_device_name(),
        seed=protocol['training']['seed'],scheduler=None,cudnn_deterministic=True,
        cudnn_benchmark=False,deterministic_algorithms=False,CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    model=optimizer=None;step=0;start=time.perf_counter();checkpoints=[]
    old_nan=torch.nan_to_num
    try:
        assert sha256(protocol['original_protocol'])==protocol['original_protocol_sha256']
        assert sha256(prepared/'manifest.json')==protocol['prepared_manifest_sha256']
        assert len(frames)==8 and sum(f['retained_gt'] for f in frames)==335
        for f,p in zip(frames,protocol['frames']):assert all(f[k]==v for k,v in p.items())
        random.seed(result['seed']);np.random.seed(result['seed']);torch.manual_seed(result['seed']);torch.cuda.manual_seed_all(result['seed'])
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(False)
        torch.cuda.set_per_process_memory_fraction(22*1024**3/torch.cuda.get_device_properties(0).total_memory)
        def checked_nan(x,*a,**kw):old.finite(x,'BeforeOfficialNanToNum');return old_nan(x,*a,**kw)
        torch.nan_to_num=checked_nan
        sys.addaudithook(old.input_guard)
        old.PERCEPTION_ACTIVE=True;blocked=0
        for canary in [prepared/'supervision/00.npz',prepared.parent/'data/samples/LIDAR_TOP/forbidden.bin']:
            try:canary.read_bytes()
            except PermissionError:blocked+=1
        old.PERCEPTION_ACTIVE=False;assert blocked==2
        archive=rt/'artifacts/safe_extract/model_tensors.npz'
        assert sha256(archive)==cfg['official']['safe_tensor_archive_sha256']
        model=old.build(rt,cfg,archive);original=snapshot(model.state_dict())
        assert old.state_sha(original)==json.loads(Path('reports/m8j6_sources/actual_baseline/result.json').read_text())['initial_state_sha256']
        tr=protocol['training']
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=tr['lr'],betas=tuple(tr['betas']),weight_decay=tr['weight_decay'])
        assert len(optimizer.state)==0
        mapping=parameter_mapping(model,optimizer,head_policy)
        assert mapping['canonical_trainable_count']==141 and len(mapping['frozen_state_names'])==337
        training_mode(model)
        coder=model.pts_bbox_head.bbox_coder
        assert coder.max_num==300 and coder.score_threshold is None
        assert list(coder.post_center_range)==protocol['evaluation']['decoder_post_center_range']
        old.save(report/'parameter_mapping.json',mapping)
        result.update(original_weight_sha256=cfg['official']['weights_sha256'],safe_archive_sha256=sha256(archive),
            original_state_sha256=old.state_sha(original),fresh_optimizer_initial_entries=0,
            input_canaries_rejected=blocked,
            dropout_modules={n:float(m.p) for n,m in model.pts_bbox_head.named_modules() if isinstance(m,torch.nn.Dropout)},
            attention_dropout={n:float(m.dropout) for n,m in model.pts_bbox_head.named_modules() if isinstance(m,torch.nn.MultiheadAttention)})
        if phase=='verify':
            cp=json.loads((report.parent/'train/checkpoint_400.json').read_text());path=Path(cp['path'])
            assert sha256(path)==cp['sha256']
            saved=torch.load(path,map_location='cpu')
            assert saved['code_commit']==commit and saved['run_id']==protocol['run_id'] and saved['global_step']==400
            assert saved['protocol_sha256']==sha256('configs/m8j7_protocol.json')
            assert saved['original_state_sha256']==old.state_sha(original) and saved['scheduler'] is None
            assert_exact(mapping,saved['mapping'],'MappingBeforeLoad')
            assert fingerprint(saved['model'])==cp['model_fingerprint']
            assert fingerprint(saved['optimizer'])==cp['optimizer_fingerprint']
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
            assert_exact(saved['model'],model.state_dict(),'ModelRestore')
            assert_exact(saved['optimizer'],optimizer.state_dict(),'OptimizerRestore')
            assert_exact(mapping,parameter_mapping(model,optimizer,head_policy),'SharedMappingAfterLoad')
            for n,m in model.named_modules():m.train(saved['module_modes'][n])
            restore_rng(saved['rng'])
            output=tensor_output_compare(saved['fixed_eval_outputs'],evaluation_probe(model,prepared,frames),1e-6,1e-5)
            assert_exact(saved['rng'],rng_state(),'RNGAfterReloadProbe')
            assert_frozen_unchanged(model,original,mapping['frozen_state_names'])
            result.update(status='FinalReloadPassed',checkpoint_sha256=cp['sha256'],global_step=400,
                model_buffers_optimizer_exact=True,shared_mapping_preserved=True,RNG_exact=True,
                fixed_eval_output_comparison=output,optimizer_state=optimizer_summary(optimizer),
                frozen_final_unchanged=True)
        else:
            evaluate(model,prepared,manifest,protocol,private,report,0)
            checkpoints.append(save_checkpoint(model,optimizer,prepared,frames,mapping,original,0,private,report,protocol,commit))
            steps_dir=private/'steps';steps_dir.mkdir(exist_ok=False)
            public_steps=report/'steps.jsonl'
            with public_steps.open('x') as stream:
                for step in range(1,401):
                    assert step<=tr['maximum_optimizer_steps']==400
                    index=(step-1)%8
                    assert not model.img_backbone.training and model.pts_bbox_head.training
                    s=optimizer_step(model,optimizer,prepared,frames[index],step-1,original,mapping,steps_dir,cfg)
                    record={k:v for k,v in s.items() if k not in ('parameter_gradient_norms','trainable_parameter_changes')}
                    record['parameter_delta_l2']=sum(x['delta_l2']**2 for x in s['trainable_parameter_changes'])**.5
                    record['private_detail_sha256']=sha256(steps_dir/f'step_{step}.json')
                    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                    result['optimizer_steps']=step
                    if step in protocol['evaluation']['steps']:
                        evaluate(model,prepared,manifest,protocol,private,report,step)
                        checkpoints.append(save_checkpoint(model,optimizer,prepared,frames,mapping,original,step,private,report,protocol,commit))
            assert len(list(steps_dir.glob('optimizer_step_*_returned.json')))==400
            assert_frozen_unchanged(model,original,mapping['frozen_state_names'])
            assert sha256(archive)==cfg['official']['safe_tensor_archive_sha256']
            result.update(status='Completed400Steps',global_step=400,frozen_final_unchanged=True,
                checkpoints=checkpoints,optimizer_state=optimizer_summary(optimizer))
    except Exception as exc:
        ledgers=list((private/'steps').glob('optimizer_step_*_returned.json'))
        result.update(status='FailedStopped',optimizer_steps=len(ledgers),error_type=type(exc).__name__,error=str(exc),checkpoints=checkpoints)
        if model is not None:
            path=private/f'failure_{phase}.pth'
            torch.save(dict(model=snapshot(model.state_dict()),optimizer=snapshot(optimizer.state_dict()) if optimizer else None,
                optimizer_steps=len(ledgers),rng=rng_state()),path)
            result['failure_checkpoint']=dict(path=str(path),sha256=sha256(path))
        raise
    finally:
        torch.nan_to_num=old_nan
        result.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start)
        old.save(report/'result.json',result)


if __name__=='__main__':main()
