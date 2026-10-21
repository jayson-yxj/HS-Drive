"""Shared expanded data and cached RGB-only model loading."""
import json
from pathlib import Path
import torch
from tools.m5f_common import input_path
from tools.m5g_common import load_pixel
from models.m5i_visual import MobilePixel
from models.m5f_pixel import ForegroundModel
from simulator.recorder import atomic_json,file_hash
ROOT='/root/gpufree-data/hs-drive-data/processed/m5i_pixels'
TEST='/root/gpufree-data/hs-drive-data/processed/m5i_test_pixels'
def rows(root=ROOT):return json.loads(Path(root,'index.json').read_text())['records']
def load(record):
    if record['name']=='mobile_gt':
        assert file_hash(record['checkpoint'])==record['sha256'];ck=torch.load(record['checkpoint'],weights_only=True);model=MobilePixel().cuda().eval();model.load_state_dict(ck['model']);return model,ck
    net,head,ck=load_pixel(record);return ForegroundModel(net.encoder,head,8).eval(),ck

def main():
    old=rows('/root/gpufree-data/hs-drive-data/processed/m5h_pixels');new=rows('/root/gpufree-data/hs-drive-data/processed/m5i_added_pixels');out=Path(ROOT);out.mkdir(exist_ok=False);records=[dict(r,input=str(input_path(r)))for r in old+new];assert len(set(r['sample_id']for r in records))==len(records);atomic_json(out/'index.json',dict(records=records,source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
