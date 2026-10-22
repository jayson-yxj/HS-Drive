"""Process one geometry-supported pilot before authorizing dataset expansion."""
import json,time
from pathlib import Path
from tools.m5o_common import DATA
from tools.m5o_campaign import run

def main():
    raw=DATA/'raw/m5o_v3_train/Town01_m5o_train_00'
    while not (raw/'manifest.json').exists():time.sleep(2)
    out=DATA/'processed/m5o_smoke_v3_index'
    run('tools.m5o_index',['--raw',str(raw),'--output',str(out)],'index_smoke_v3.log')
    pix=DATA/'processed/m5o_smoke_v3_pixels'
    run('tools.m5g_dataset',['--indices',str(out),'--output',str(pix),'--splits','train'],'pixels_smoke_v3.log')
    run('tools.m5o_cache',['--split','smoke_v3'],'cache_smoke_v3.log','.venv-gpu/bin/python')
    run('tools.m5o_candidate_audit',['--split','smoke_v3'],'candidate_smoke_v3.log','.venv-gpu/bin/python')
if __name__=='__main__':main()
