"""Reproducible image evidence sheets; assistant review is separate from human certification."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from tools.m8c_common import *
from datasets.m8b_petr_adapter import corners_hs
OUT=Path('reports/m8c_figures')
def save(fig,name):
    p=OUT/(name+'.png');assert not p.exists();fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig)
def pane(ax,record,view):
    p=json.loads((BASELINE/'rgb_bundle'/record['token']/'packet.json').read_text());ci=view['camera_index'];rgb=np.asarray(Image.open(p['cameras'][ci]['path']).convert('RGB'));ax.imshow(rgb)
    b=view.get('projected_bbox',view.get('bounds'));g=view.get('instance_bbox')
    if b is not None:ax.add_patch(Rectangle(b[:2],b[2]-b[0],b[3]-b[1],fill=False,color='red',lw=1.5))
    if g is not None:ax.add_patch(Rectangle(g[:2],g[2]-g[0],g[3]-g[1],fill=False,color='lime',lw=1.5))
    if b is not None:
        both=[b]+([g] if g is not None else []);bb=np.array(both);x0,y0=bb[:,:2].min(0)-25;x1,y1=bb[:,2:].max(0)+25;ax.set_xlim(max(0,x0),min(640,x1));ax.set_ylim(min(384,y1),max(0,y0))
    ax.set_title(record['token']+' q'+str(record['prediction_index'])+' '+view['camera']+'\n'+record.get('grade',record.get('status',''))+' s=%.3f'%record['score'],fontsize=8);ax.axis('off')
def main():
    OUT.mkdir(exist_ok=False);data=json.loads((ROOT/'association_ledger.json').read_text());assoc=data['associations'];supported=[r for r in assoc if r['grade']=='IdentitySupported'];far=[r for r in assoc if 20<=r['range_m']<40 and r['views']][:8]
    for name,rows in [('identity_supported',supported),('far_proxy',far)]:
        fig,axes=plt.subplots(len(rows),2,figsize=(12,len(rows)*2.2),squeeze=False)
        for row,ax in zip(rows,axes):
            vv=sorted(row['views'],key=lambda v:-v['instance_bbox_iou'])[:2]
            for a,v in zip(ax,vv):pane(a,row,v)
            if len(vv)<2:ax[1].axis('off')
        fig.suptitle('Red: candidate projection; green: tight rendered instance box. Evidence grade is NOT human-certified identity.',fontsize=10);fig.tight_layout();save(fig,name)
    bg=data['background'];pages=[]
    for start in range(0,len(bg),12):
        chunk=bg[start:start+12];fig,axes=plt.subplots(4,3,figsize=(15,12))
        for ax,row in zip(axes.flat,chunk):
            if row['views']:pane(ax,row,max(row['views'],key=lambda v:v['area']))
            else:ax.text(.1,.5,row['token']+' no supported visible projection');ax.axis('off')
        for ax in list(axes.flat)[len(chunk):]:ax.axis('off')
        name='background_%02d'%(start//12);fig.suptitle('Frozen background audit: ALL sampled candidates. Red is projected 3D hypothesis, not confirmed object.',fontsize=11);fig.tight_layout();save(fig,name);pages.append(dict(file=str(OUT/(name+'.png')),candidates=[dict(token=r['token'],prediction_index=r['prediction_index']) for r in chunk]))
    fig,axes=plt.subplots(2,2,figsize=(12,9));support=data['support']
    for c,col in [('vehicle','blue'),('person','orange'),('cone','green')]:
        ss=[s for s in support if s['type']==c];axes[0,0].scatter([s['range_m'] for s in ss],[s['native_pixels'] for s in ss],s=10,label=c,color=col);axes[0,1].scatter([s['range_m'] for s in ss],[s['native_visible_height_px'] for s in ss],s=10,label=c,color=col)
    axes[0,0].set(yscale='log',xlabel='GT range m',ylabel='Native visible instance pixels');axes[0,0].legend();axes[0,1].set(xlabel='GT range m',ylabel='Retained native visible height px');axes[0,1].legend()
    hook=json.loads((ROOT/'hook/run.json').read_text())['records'][0]
    for c in hook['camera']:axes[1,0].plot(hook['depth_samples'],c['outside_by_depth'],label=c['camera'])
    axes[1,0].set(xlabel='PETR geometry sampling optical depth m',ylabel='Fraction outside position_range');axes[1,0].legend(fontsize=7)
    grades=['IdentitySupported','ProjectionProxyOnly','AmbiguousAssociation','NoSupportedAssociation'];tot=[sum(r['grade']==g for r in assoc) for g in grades];farcounts=[sum(r['grade']==g and 20<=r['range_m']<40 for r in assoc) for g in grades];x=np.arange(4);axes[1,1].bar(x-.18,tot,.36,label='all155');axes[1,1].bar(x+.18,farcounts,.36,label='far83');axes[1,1].set_xticks(x);axes[1,1].set_xticklabels(['Identity','Proxy','Ambiguous','No pair']);axes[1,1].legend();axes[1,1].set_ylabel('Known objects');fig.tight_layout();save(fig,'support_position_grades')
    write('reports/m8c_visualization_manifest.json',dict(**provenance('OfflineDiagnosticSheets'),files=[asset(p) for p in sorted(OUT.glob('*.png'))],background_pages=pages,identity_selection='All6automaticallysupported; first8farproxies in frozen order. Notmodelselection.',human_certified=False))
if __name__=='__main__':main()
