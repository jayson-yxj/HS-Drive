"""Stream one completed training group for ground/pair/supervision health only."""
import argparse,json
from pathlib import Path
from tools.m5o_common import DATA,write,provenance
from tools.m5o_campaign import run

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);a=p.parse_args()
    if '_train_' not in a.group:raise ValueError('debug audit restricted to Train')
    raw=DATA/'raw/m5o_v3_train'/a.group;out=DATA/'processed'/f'm5o_audit_{a.group}_index';pix=DATA/'processed'/f'm5o_audit_{a.group}_pixels'
    run('tools.m5o_index',['--raw',str(raw),'--output',str(out)],f'audit_index_{a.group}.log')
    run('tools.m5g_dataset',['--indices',str(out),'--output',str(pix),'--splits','train'],f'audit_pixels_{a.group}.log')
    from collections import Counter
    rows=json.loads((pix/'index.json').read_text())['records'];support=Counter();ambiguous=[]
    for r in rows:
        for cam in r['actors']:
            for actor in cam['actors']:
                if actor['visible_eligible']:support[cam['camera']]+=1
    write(f'reports/m5o_capture_support_{a.group}.json',dict(kind='offline_actual_CARLA_train_only',rows=len(rows),visible_views=dict(support),provenance=provenance()))
if __name__=='__main__':main()
