"""M5-L immutable study outputs and data split identity."""
import json
from pathlib import Path
from tools.m5k_common import write,asset,summary,group_depth,ROOT
from tools.m5k_common import rows as krows

def rows(split):
    if split in ['train','val']:return krows(split)
    if split=='regression':return krows('test')
    if split=='test':return json.loads((ROOT/'m5l_test_pixels/index.json').read_text())['records']
    raise ValueError(split)

def cache_index(split):
    old='test'if split=='regression'else split
    path=Path(f'runs/m5k_alignment_{old}/raw.json')if split!='test'else Path('runs/m5l_baseline_test/raw.json')
    return {r['sample_id']:r['path']for r in json.loads(path.read_text())['index']}
