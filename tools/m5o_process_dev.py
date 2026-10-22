"""Gate-driven processing; waits for campaign completion, then finite research only."""
import json,time,subprocess
from pathlib import Path
from tools.m5o_campaign import run
from tools.m5o_common import DATA,write,asset,provenance
from simulator.recorder import file_hash


def main():
    for split in ['train','val','calibration']:
        path=DATA/'processed'/f'm5o_{split}_pixels/index.json'
        while not path.exists():
            if Path('runs/m5o_logs/campaign_dev.log').exists() and 'Traceback' in Path('runs/m5o_logs/campaign_dev.log').read_text():raise RuntimeError('capture failed; preserve partial assets')
            time.sleep(5)
        run('tools.m5o_cache',['--split',split],f'cache_{split}.log','.venv-gpu/bin/python')
    run('tools.m5o_support',[],'support.log','.venv-gpu/bin/python')
    if not json.loads(Path('reports/m5o_dataset_support.json').read_text())['passed']:
        write('reports/m5o_training.json',dict(status='NotRun',reason='independent support gate failed'));return
    run('tools.m5o_train',[],'training.log','.venv-gpu/bin/python')
    run('tools.m5o_evaluate',['--split','val'],'eval_val.log','.venv-gpu/bin/python')
    run('tools.m5o_evaluate',['--split','calibration'],'eval_calibration.log','.venv-gpu/bin/python')
    lock=provenance()['sources'];lock.update({p:file_hash(p) for p in ['configs/m5o_protocol.json','reports/m5o_training.json','reports/m5o_calibration_lock.json']})
    training=json.loads(Path('reports/m5o_training.json').read_text())
    lock.update({r['selected']['path']:r['selected']['sha256'] for r in training['records']})
    write('reports/m5o_pretest_freeze.json',dict(assets=lock,provenance=provenance(),protocol=asset('configs/m5o_protocol.json'),no_test_selection=True))
if __name__=='__main__':main()
