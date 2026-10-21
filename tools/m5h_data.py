"""Combine immutable prior training data with new train-only coverage scenes."""
import json
from pathlib import Path
from tools.m5f_common import input_path
from simulator.recorder import atomic_json,file_hash
ROOT='/root/gpufree-data/hs-drive-data/processed/m5h_pixels'
def rows(root=ROOT):return json.loads(Path(root,'index.json').read_text())['records']
def main():
    old=rows('/root/gpufree-data/hs-drive-data/processed/m5f_pixels');new=rows('/root/gpufree-data/hs-drive-data/processed/m5h_added_pixels');out=Path(ROOT);out.mkdir(exist_ok=False)
    assert all(r['split']=='train' for r in new)
    records=[dict(r,input=str(input_path(r)))for r in old+new]
    assert len({r['sample_id']for r in records})==len(records)
    assert not({r['group']for r in records if r['split']=='train'}&{r['group']for r in records if r['split']=='val'})
    atomic_json(out/'index.json',dict(records=records,source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
