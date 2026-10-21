"""Train/Val teacher disagreement examples; diagnostic selection never changes masks."""
import json
from pathlib import Path
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m5i_data import rows
from tools.m5i_teacher_audit import teacher_index
from tools.m5f_common import pixel_label,input_path
from simulator.recorder import atomic_json,file_hash

def main():
    out=Path('reports/m5i_teacher_audit_figures');out.mkdir(exist_ok=False);rr={r['sample_id']:r for r in rows()};idx=teacher_index();records=json.loads(Path('reports/m5i_teacher_transfer_audit.json').read_text())['records'];chosen=[]
    for key in ['gt_negative_teacher_conflict','gt_positive_teacher_miss']:
        seen=set()
        for r in sorted([r for r in records if r['split']=='val'],key=lambda r:r[key],reverse=True):
            pair=(r['sample_id'],r['camera'])
            if pair in seen:continue
            seen.add(pair);chosen.append((key,r))
            if len(seen)==2:break
    meta=[]
    for j,(key,r)in enumerate(chosen):
        row=rr[r['sample_id']];d=pixel_label(row);c=row['camera_names'].index(r['camera']);inp=json.loads(input_path(row).read_text());raw=Path(inp['images'][c]['path']);im=cv2.cvtColor(cv2.imread(str(raw)),cv2.COLOR_BGR2RGB);q=np.load(idx[r['sample_id']]['path'])['score'].max(0)[c]/256;fig,ax=plt.subplots(1,3,figsize=(15,4));ax[0].imshow(im);ax[0].set_title(r['group']+' '+r['camera']);ax[1].imshow(np.where(d['valid'][c],d['target'][c].astype(float),np.nan),vmin=0,vmax=1);ax[1].set_title('Reliable GT / unknown white');ax[2].imshow(q,vmin=0,vmax=1,cmap='magma');ax[2].set_title('Teacher RGB score');fig.suptitle(key+'; not independent mesh adjudication')
        for a in ax:a.axis('off')
        fig.tight_layout();p=out/f'm5i_teacher_disagreement_{j}.png';fig.savefig(p,dpi=140);plt.close(fig);meta.append(dict(selected_by=key,record=r,path=str(p),sha256=file_hash(p)))
    atomic_json('reports/m5i_teacher_audit_visualizations.json',dict(kind='offline_trainval_disagreement_visualization',records=meta,source_sha256=file_hash(__file__),limits='Diagnostic maxcounts on validation; no test, no mask/rule modification; disagreement does not independently adjudicate renderer versus teacher.'))
if __name__=='__main__':main()
