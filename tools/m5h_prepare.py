"""Freeze prior assets, review complete JSON documents, predeclare M5H study."""
import json, subprocess
from pathlib import Path
from simulator.recorder import atomic_json, file_hash

def main():
    out=Path('reports/m5h_baseline_freeze.json')
    if out.exists():raise FileExistsError(out)
    required=['summary.md','acceptance.json','pretrained_2d_baseline.json','pixel_precision_recall.json','training_exposure.json','depth_training.json','depth_spatial_quality.json','oracle_sampling_ceiling.json']
    review={}
    for name in required:
        p=Path('reports/m5g_'+name);content=p.read_text();value=json.loads(content) if p.suffix=='.json' else content
        review[str(p)]=dict(sha256=file_hash(p),bytes=p.stat().st_size,top_level=list(value) if isinstance(value,dict) else 'markdown')
    tracked=subprocess.check_output(['git','ls-files'],text=True).splitlines()
    prior=json.loads(Path('reports/m5g_artifact_manifest.json').read_text())
    assets={p:v['sha256'] for p,v in prior['assets'].items()}
    assets.update(json.loads(Path('reports/m5g_baseline_freeze.json').read_text())['checkpoint_sha256'])
    atomic_json(out,dict(base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),tracked_sha256={p:file_hash(p) for p in tracked},assets_sha256=assets,review=review,findings=['Zero rear_right train positives; sampler omitted 500+ views at1200','Teacher supported vehicles/person only; native384x640 internally800x1333','All trained M5G Depth rejected near5percent gate','Renderer reference is not independentMesh; hist row0positive row1negative']))
    atomic_json('configs/m5h_protocol.json',dict(schema='hsd-m5h-study-v1',base='bf0c38d',seeds=[101,102],pixel=dict(models=['gt','distill'],steps=2400,evaluate_every=400,head_lr=.001,fpn_lr=.0001,teacher_lambda=.25,teacher_confidence=.7,teacher_pixel_min=.7,teacher_domain='Only reliable GT vehicle/person positives with agreeing supported teacher mask>=.7 and instance confidence>=.7. No teacher negative supervision; uncertain/conflicting/unsupported pixels remain GT-only.',budget_reason='Predeclared2400 from start: added16trainframes give1248 eligible six-camera views;1200 cannot cover one pass. No conditional extension beyond2400.',sampler='First shuffled epoch covers ALL eligible train views once; remaining steps inverse-frequency multi-stratum weighted sampling, all arms identical.',selection='Validation only sceneP>=.8 R>=.8 EmptyFPR<=.01 minimax thenmacroIoU thenearlieststep',conditional_hard='Only if BOTH distill seeds improve validation macroAP by>=.01 and jointviolation not worse, and background FPR>0.001; no automatic budget extension',backbone='No change; do not attribute residual gap uniquely tocapacity'),depth=dict(models=['original','replay','replay_kl'],steps=1200,evaluate_every=200,lr=.00002,replay_weight=1.,kl_weight=10.,bins=48,new_loss='Uniform reliable-ray CE; no equal-distance or actor upweighting',replay='All original M4 expanded_joint train_sample_ids, original S16 and diagnostic P8 near1.5-10m reliable depth, no original validation/test.',kl='old||new on reliable near rays from newdata plusreplay, FP32,mean over rays',selection='Nonzero step nearMAE<=1.05old on currentValP8 AND originalValS16; maximizeVal10-35 target1mmass',conditional_spatial='Only if BOTH B and C have nonzero eligible checkpoints and no globalMAE degradation>5percent; otherwise D not run'),test=dict(groups=6,role='New poses aftergeometryallocation, capture/open onlyaftermodelthresholdfreeze',distance_min_m=45),shadow=dict(threshold=.2,student_gt_mask=False,oracle='diagnostic only',no_control=True),roi=dict(stride=4,selection='student probability at validation chosen pixelthreshold',depth='Oracle only sampling diagnosis, not implementedS4Depthnetwork'),low_fpr=[.01,.001,.0001],gates='Originalquality gates unchanged; runtime inheritedPartial; purevisualsafety notvalidated'))
if __name__=='__main__':main()
