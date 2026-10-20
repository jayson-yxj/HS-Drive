"""Synthetic acceptance evidence: failed checks never silently close M4."""
import json
import subprocess
import sys
from pathlib import Path
from simulator.recorder import file_hash


def test_baseline_gate_rejects_failed_resume_and_different_checkpoint(tmp_path):
    run=tmp_path/'run';fit=tmp_path/'fit';run.mkdir();fit.mkdir()
    def write(path,value):path.write_text(json.dumps(value))
    (run/'best.pt').write_bytes(b'synthetic fixture, not a usable model')
    digest=file_hash(run/'best.pt')
    record={'status':'complete','all_gradients_finite':True,'peak_allocated_gib':1.,
            'checkpoint_reload_exact':True,'best_sha256':digest,'index_sha256':'same','train_sample_ids':['train/1']}
    write(run/'manifest.json',record);write(fit/'manifest.json',record)
    write(fit/'initial_train.json',{'loss':{'depth_ce':4.,'surface_mse':1.}})
    write(fit/'final_train.json',{'loss':{'depth_ce':1.,'surface_mse':.1}})
    gpu=tmp_path/'gpu.json';resume=tmp_path/'resume.json'
    write(gpu,{'passed':True,'checks':{'bf16_supported':True}});write(resume,{'passed':False})
    for split in ('val','test'):
        write(tmp_path/f'{split}.json',{'split':split,'samples':1,'checkpoint_sha256':digest,
            'index_sha256':'same','sample_ids':[split+'/1'],'inference_without_supervision':True,
            'output_shapes':{'bev_current':[1,128,160,120],'height_bev':[1,64,4,160,120]}})
    output=tmp_path/'gate.json'
    command=[sys.executable,'-m','tools.validate_m4_baseline','--run',str(run),'--overfit',str(fit),
             '--eval-val',str(tmp_path/'val.json'),'--eval-test',str(tmp_path/'test.json'),
             '--gpu',str(gpu),'--resume',str(resume),'--output',str(output)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==1,result.stderr
    gate=json.loads(output.read_text());assert not gate['passed'] and not gate['checks']['resume']
    assert not gate['full_m4_closed']
    write(resume,{'passed':True});(run/'best.pt').write_bytes(b'changed')
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==1,result.stderr
    assert not json.loads(output.read_text())['checks']['same_selected_checkpoint']
