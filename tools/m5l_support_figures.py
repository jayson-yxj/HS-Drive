"""Actual pedestrian silhouette examples selected without model-error tuning."""
import json
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.m5l_common import rows,write,asset
from tools.m5f_common import inputs,pixel_label

def main():
 audit=json.loads(Path('runs/m5l_sampling_actor_trace.json').read_text())['records'];outputs=[]
 for split in ['train','val']:
  chosen=next(r for r in audit if r['split']==split and r['band']=='20_35'and r['type']=='person'and r['associated']>0);row=next(r for r in rows(split)if r['sample_id']==chosen['sample_id']);ci=row['camera_names'].index(chosen['camera']);d=pixel_label(row);x=inputs(row,'cpu');m=d['actor'][ci]==chosen['local_id'];v,u=np.where(m);x0=max(0,int(u.min())-12);x1=min(640,int(u.max())+13);y0=max(0,int(v.min())-12);y1=min(384,int(v.max())+13);fig,axes=plt.subplots(1,3,figsize=(10,5))
  for ax,stride,phase in zip(axes,[1,8,4],[0,4,2]):
   ax.imshow(x['images'][0,ci].permute(1,2,0).numpy());ax.contour(m,levels=[.5],colors=['yellow'],linewidths=1)
   if stride>1:
    yy,xx=np.mgrid[phase:384:stride,phase:640:stride];keep=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1);ax.scatter(xx[keep],yy[keep],s=15,color='cyan',marker='+')
   ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_title('Reliableinterior'if stride==1 else f'UV phase{phase} stride{stride}')
  fig.suptitle(f'{split}: {chosen["group"]} {chosen["camera"]}');fig.tight_layout();path=Path('reports/m5l_figures')/f'm5l_{split}_pedestrian_support.png';fig.savefig(path,dpi=170);plt.close(fig);outputs.append(dict(selection='first20-35massociatedpedestrian',sample_id=row['sample_id'],actor_id=chosen['actor_id'],**asset(path)))
 write('reports/m5l_sampling_visualizations.json',dict(kind='actual_recordedRGB_label_overlay',records=outputs,limits='Yellowreliableinterior notfull silhouette. Cyan fixedraycenters, not learnedfeature activation. No sample chosenbasedonpredictionerror.'))
if __name__=='__main__':main()
