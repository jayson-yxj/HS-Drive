"""Read-only M5-I GT baseline on frozen new data; no influence on selection."""
import json
from pathlib import Path
import numpy as np,torch
from tools.m5j_common import rows,write
from tools.m5f_common import inputs,pixel_label
from tools.m5i_data import load
from tools.m5g_evaluation import summarize
from simulator.recorder import file_hash
from training.runtime import prepare_cuda

def main():
    from tools.m5j_freeze import verify
    verify();prepare_cuda();torch.set_num_threads(2);study=json.loads(Path('reports/m5i_distillation_comparison.json').read_text())['records'];rr=rows('test');labels=[pixel_label(r)for r in rr];records=[]
    for rec in study:
        if rec['name']!='gt':continue
        model,ck=load(rec);probs=[]
        for row in rr:
            x=inputs(row)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):q=np.minimum(model(**x).float().sigmoid()[0].cpu().numpy()*256,255).astype(np.uint8)
            probs.append(q.astype(np.float32)/256)
        result=summarize(rr,labels,probs,ck['selection']['threshold']);records.append(dict(name=rec['name'],seed=rec['seed'],checkpoint=rec['checkpoint'],checkpoint_sha256=rec['sha256'],result=result))
    write('reports/m5j_matched_resnet_regression.json',dict(kind='postfreeze_readonly_historical_M5IGT_matchedbudget_baseline',records=records,source_sha256=file_hash(__file__),limits='No newmodel/thresholdchoice; originalM5IselectedGT checkpoint andthreshold fixedhistorically. Descriptive supplement toM5Glegacy; no changeM5Jdecisions.'))
if __name__=='__main__':main()
