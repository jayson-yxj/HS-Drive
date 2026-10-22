"""Single sealed capture/evaluation after immutable model and calibration lock."""
import json
from pathlib import Path
from tools.m5o_campaign import run
from tools.m5o_common import write,asset,provenance
from simulator.recorder import file_hash


def main():
    freeze=json.loads(Path('reports/m5o_pretest_freeze.json').read_text())
    for p,h in freeze['assets'].items():assert file_hash(p)==h,p
    run('tools.m5o_allocate',['--phase','sealed'],'allocate_sealed.log')
    run('tools.m5o_road_layouts',['--phase','sealed'],'road_layouts_sealed.log')
    run('tools.m5o_campaign',['--phase','sealed'],'campaign_sealed.log')
    run('tools.m5o_cache',['--split','sealed'],'cache_sealed.log','.venv-gpu/bin/python')
    run('tools.m5o_evaluate',['--split','sealed'],'eval_sealed.log','.venv-gpu/bin/python')
    run('tools.m5o_candidate_audit',['--split','sealed'],'candidate_sealed.log','.venv-gpu/bin/python')
if __name__=='__main__':main()
