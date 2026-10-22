"""Resumable by complete group; partial runs preserved, never silently overwritten."""
import argparse,json,subprocess,sys
from pathlib import Path
from tools.m5o_common import DATA,write


def run(module,args,log,python='.venv/bin/python'):
    path=Path('runs/m5o_logs')/log
    with path.open('x') as f:
        subprocess.run([python,'-m',module,*args],stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','sealed'],required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    protocol=f'configs/m5o_capture_{a.phase}_v3.json';groups=json.loads(Path(protocol).read_text())['groups']
    if a.limit:groups=groups[:a.limit]
    for g in groups:
        split=g['split'];parent=DATA/'raw'/f'm5o_v3_{split}';stage=parent/g['group']
        if (stage/g['group']/'manifest.json').exists():continue
        # Each capture owns its root, nested intentionally for append-only resume.
        run('tools.m5o_capture',['--protocol',protocol,'--groups',g['group'],'--output',str(stage)],f"capture_v3_{g['group']}.log")
        print('captured',g['group'],flush=True)
    if a.limit:return
    for split in sorted({g['split'] for g in groups}):
        # Symlinks give downstream immutable single-level index, no moving raw files.
        linked=DATA/'raw'/f'm5o_{split}_indexroot';linked.mkdir(exist_ok=True)
        for g in groups:
            if g['split']!=split:continue
            target=DATA/'raw'/f'm5o_v3_{split}'/g['group']/g['group'];link=linked/g['group']
            if not link.exists():link.symlink_to(target,target_is_directory=True)
        out=DATA/'processed'/f'm5o_{split}_index'
        if not out.exists():run('tools.m5o_index',['--raw',str(linked),'--output',str(out)],f'index_{split}.log')
        pix=DATA/'processed'/f'm5o_{split}_pixels'
        if not pix.exists():run('tools.m5g_dataset',['--indices',str(out),'--output',str(pix),'--splits',split],f'pixels_{split}.log')
        print('processed',split,flush=True)

if __name__=='__main__':main()
