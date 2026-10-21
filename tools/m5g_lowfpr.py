"""Validation-only low-FPR operating point, evaluated once without test optimization."""
import json
from pathlib import Path
import numpy as np
from diagnostics.m5f_metrics import metrics
from simulator.recorder import atomic_json,file_hash


def main():
    out=Path('reports/m5g_lowfpr_workpoints.json')
    if out.exists():raise FileExistsError(out)
    val=json.loads(Path('reports/m5g_pixel_val.json').read_text())['records'];test=json.loads(Path('reports/m5g_pixel_test.json').read_text())['records'];records=[]
    for r in val:
        h=np.array(r['result']['histogram']);curve=[dict(threshold=t/100,**metrics(h,t/100))for t in range(1,100)];eligible=[x for x in curve if(x['fpr']or 0)<=.001];selected=max(eligible,key=lambda x:(x['recall']or 0,-x['threshold']))if eligible else None;t=next(x for x in test if x['name']==r['name']and x['seed']==r['seed']);records.append(dict(name=r['name'],seed=r['seed'],val_selected=selected,test_metrics=metrics(np.array(t['result']['histogram']),selected['threshold'])if selected else None))
    atomic_json(out,dict(kind='offline_validation_chosen_secondary_operating_point',records=records,source_sha256=file_hash(__file__),limits='SecondaryFPR.001 threshold derivedonlysavedVal; calculatedposttest but notusedformodelselectionorprimaryresults; notpredeclaredacceptancegate. Largebackgrounddenominator mayhide localresidue.'))

if __name__=='__main__':main()
