"""Evaluator-side selection/export. Inference receives only sanitized RGB packets."""
import json,shutil,subprocess
from pathlib import Path
from collections import Counter
from tools.m8b_common import CFG,ROOT,RUNTIME,LOG,provenance,asset,sha,write
from tools.m5o_common import rows
from tools.m7a_shadow_prepare import export

def main():
    ROOT.mkdir(exist_ok=False);(ROOT/'home').mkdir();bundle=ROOT/'rgb_bundle';bundle.mkdir()
    previous=json.loads(Path('reports/m8a_p1_release_manifest.json').read_text());bad=[a['path'] for a in previous['files'] if sha(a['path'])!=a['sha256']];assert not bad
    ck=Path('/root/gpufree-data/hs-drive-experiments/m8a-proxy-retry/assets/epoch_24.pth');assert sha(ck)==CFG['checkpoint_sha256']
    safe=RUNTIME/'artifacts/safe_extract/model_tensors.npz';assert sha(safe)==CFG['numeric_weights_sha256']
    allrows=[r for s in ['train','val','calibration','sealed'] for r in rows(s)];byid={r['sample_id']:r for r in allrows}
    core=json.loads(Path('reports/m7a_six_camera_shadow_contract.json').read_text())['records'];selected=[byid[r['sample_id']] for r in core];coreids={r['sample_id'] for r in selected}
    for g in sorted({r['group'] for r in allrows}):
        eligible=sorted([r for r in allrows if r['group']==g and any(a['visible_eligible'] for c in r['actors'] for a in c['actors'])],key=lambda r:(r['configuration'],r['sample_id']))
        for q in CFG['sampling']['extension_quantiles']:
            if eligible:selected.append(eligible[round(q*(len(eligible)-1))])
    selected=list({r['sample_id']:r for r in selected}.values());assert len(selected)<=200
    packets=[];index=[];audits=[]
    for i,row in enumerate(selected):
        p,errors=export(row);token='s%03d'%i;folder=bundle/token;folder.mkdir()
        cameras=[]
        for ci,c in enumerate(p['cameras']):
            path=folder/(c['name']+'_rgb.png');shutil.copyfile(c['path'],path)
            cameras.append(dict(name=c['name'],path=str(path),sha256=c['sha256'],frame_id=c['frame_id'],timestamp_s=c['timestamp_s'],K=p['intrinsics'][ci],T_cam_to_ego=p['cam_to_ego'][ci]))
        packet=dict(schema='m8b.six_rgb.v1',token=token,camera_order=CFG['camera_order'],frame_id=p['frame_id'],timestamp_s=p['timestamp_s'],image_size=[384,640],cameras=cameras)
        file=folder/'packet.json';write(file,packet);packets.append(asset(file))
        index.append(dict(token=token,core40=row['sample_id'] in coreids,row=row,packet=asset(file)))
        audits.append(dict(token=token,group=row['group'],split=row['split'],sample_id=row['sample_id'],core40=row['sample_id'] in coreids,frame=p['frame_id'],timestamp_s=p['timestamp_s'],images=[dict(name=c['name'],sha256=c['sha256']) for c in cameras],calibration_roundtrip_max_pixel=max(errors),packet=asset(file),input_source=asset(row['input']),raw_reference_evaluator_only=asset(row['raw_frame'])))
    manifest=dict(schema='m8b.inference_manifest.v1',packets=packets,weights=dict(path=str(safe),sha256=CFG['numeric_weights_sha256']),checkpoint_sha256=CFG['checkpoint_sha256'],petr_source=str(RUNTIME/'sources/PETR'))
    write(bundle/'manifest.json',manifest);write(ROOT/'evaluation_index.json',index)
    write('reports/m8b_selection_freeze.json',dict(**provenance('PreInferenceSelectionFreeze'),frames=len(index),core_frames=40,unique_groups=len({x['row']['group'] for x in index}),split_frames=dict(Counter(x['row']['split'] for x in index)),selection_rule=CFG['sampling'],evaluation_index=asset(ROOT/'evaluation_index.json'),inference_manifest=asset(bundle/'manifest.json'),rows=audits))
    write(LOG/'baseline_freeze.json',dict(**provenance('ActualReadOnlyBaselineReverification'),historical_manifest=asset('reports/m8a_p1_release_manifest.json'),historical_mismatches=bad,checkpoint=asset(ck),numeric_weights=asset(safe),PETR_commit=subprocess.check_output(['git','-C',str(RUNTIME/'sources/PETR'),'rev-parse','HEAD'],text=True).strip()))
    print('Frozen',len(index),'frames; all40 core; positions',len({x['row']['group'] for x in index}),flush=True)
if __name__=='__main__':main()
