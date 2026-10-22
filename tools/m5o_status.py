"""Read-only campaign progress, failures and completed phase inventory."""
import json
from pathlib import Path
from tools.m5o_common import DATA

def main():
    cfg=json.loads(Path('configs/m5o_capture_dev_v3.json').read_text());out={}
    for split in ['train','val','calibration']:
        gs=[g for g in cfg['groups'] if g['split']==split];done=[];active=[]
        for g in gs:
            root=DATA/'raw'/f'm5o_v3_{split}'/g['group'];man=root/g['group']/'manifest.json'
            if man.exists():done.append(g['group'])
            elif root.exists():active.append(g['group'])
        out[split]=dict(requested=len(gs),completed=len(done),active=active,index=(DATA/'processed'/f'm5o_{split}_pixels/index.json').exists(),cache=Path(f'runs/m5o_cache_{split}/supervision.json').exists())
    for p in ['campaign_dev','process_dev','stream_index']:
        path=Path(f'runs/m5o_logs/{p}.log')
        if path.exists() and 'Traceback' in path.read_text():out[p+'_failure']=path.read_text()[-1500:]
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
