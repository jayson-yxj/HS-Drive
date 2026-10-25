"""Ground-truth evidence sheets only; heldout shows no model predictions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tools.m8e_common import *

def main():
    groups=json.loads(Path('configs/m8e/protocol.json').read_text())['groups']
    files=[];selection=[]
    for g in groups:
        audit=json.loads((ROOT/'audit'/(g['group']+'.json')).read_text())
        f=next(f for f in audit['frames'] if f['phase']=='present')
        side=json.loads(Path(f['sidecar']['path']).read_text())
        fig,axes=plt.subplots(2,3,figsize=(15,7))
        for ci,(cam,ax) in enumerate(zip(side['calibration'],axes.ravel())):
            ax.imshow(Image.open(side['sensors'][cam['name']]['rgb']['path']))
            for o in f['objects']:
                if not o.get('is_controlled_target'):continue
                for v in o['views']:
                    if v['camera_index']!=ci or not v['native_bbox']:continue
                    b=np.asarray(v['native_bbox'])
                    color='lime' if v['accepted'] else 'orange'
                    ax.add_patch(plt.Rectangle(b[:2],*(b[2:]-b[:2]),fill=False,color=color,lw=1))
                    ax.text(b[0],max(12,b[1]-2),'%s %.0fm %s'%(o['petr_class'],o['range_m'],o['entity_id']),color=color,fontsize=6)
            ax.axhline(332/2.2,color='cyan',ls='--',lw=.6);ax.set_title(cam['name']);ax.axis('off')
        fig.suptitle(g['group']+' / Present / source references only; orange = unqualified view')
        path=Path('reports/m8e_figures')/(g['group']+'.png')
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=130,bbox_inches='tight');plt.close(fig);files.append(asset(path))
        far=[o for o in f['objects'] if o.get('is_controlled_target') and 20<=o['range_m']<40 and o['identity_grade'].startswith('RenderLinked')]
        for o in far:selection.append((g,audit,o))
    for start in range(0,len(selection),6):
        part=selection[start:start+6]
        fig,axes=plt.subplots(len(part),4,figsize=(14,2.5*len(part)),squeeze=False)
        for i,(g,audit,o) in enumerate(part):
            view=max([v for v in o['views'] if v['accepted']],key=lambda v:v['native_pixels'])
            camera=view['camera'];box=np.asarray(view['native_bbox'])
            x0=max(0,int(box[0])-45);x1=min(640,int(box[2])+45);y0=max(0,int(box[1])-40);y1=min(384,int(box[3])+40)
            for j,phase in enumerate(['empty','present','shifted','removed']):
                f=next(f for f in audit['frames'] if f['phase']==phase);s=json.loads(Path(f['sidecar']['path']).read_text())
                rgb=np.asarray(Image.open(s['sensors'][camera]['rgb']['path']))
                axes[i,j].imshow(rgb[y0:y1,x0:x1]);axes[i,j].axis('off')
                axes[i,j].set_title('%s %s'%(g['group'],phase),fontsize=7)
                if j==1:axes[i,j].set_xlabel('%s %.1fm %s'%(o['petr_class'],o['range_m'],camera))
        path=Path('reports/m8e_figures')/('far_controls_%02d.png'%(start//6))
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=130,bbox_inches='tight');plt.close(fig);files.append(asset(path))
    write('reports/m8e_visualization_manifest.json',dict(**provenance('OfflineActualCapturedRGBReferenceSheets'),files=files,
        all_groups_included=True,all_qualified_far_present_objects_included=len(selection),
        heldout='Annotation review only, no model outputs',manual_identity_grade_changes=0,
        limitation='Fixed present ROI for shifted phase may move target outside crop. Controls are source evidence, not human double verification.'))
if __name__=='__main__':main()
