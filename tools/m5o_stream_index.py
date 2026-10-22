"""Index completed split while later maps capture; single writer per split."""
import json,time
from pathlib import Path
from tools.m5o_common import DATA
from tools.m5o_campaign import run


def main():
    gs=json.loads(Path('configs/m5o_capture_dev_v3.json').read_text())['groups']
    for split in ['train','val']:
        groups=[g for g in gs if g['split']==split]
        def done():return all((DATA/'raw'/f'm5o_v3_{split}'/g['group']/g['group']/'manifest.json').exists() for g in groups)
        while not done():time.sleep(5)
        linked=DATA/'raw'/f'm5o_{split}_indexroot';linked.mkdir(exist_ok=True)
        for g in groups:
            link=linked/g['group'];target=DATA/'raw'/f'm5o_v3_{split}'/g['group']/g['group']
            if not link.exists():link.symlink_to(target,target_is_directory=True)
        out=DATA/'processed'/f'm5o_{split}_index'
        if not out.exists():run('tools.m5o_index',['--raw',str(linked),'--output',str(out)],f'index_stream_{split}.log')
        pix=DATA/'processed'/f'm5o_{split}_pixels'
        if not pix.exists():run('tools.m5g_dataset',['--indices',str(out),'--output',str(pix),'--splits',split],f'pixels_stream_{split}.log')
        print('indexed',split,flush=True)
if __name__=='__main__':main()
