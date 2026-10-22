"""Complete sealed run once development lock exists, with independent audit outputs."""
import json,time
from pathlib import Path
from tools.m5o_campaign import run


def main():
    while not Path('reports/m5o_pretest_freeze.json').exists():
        p=Path('runs/m5o_logs/process_dev.log')
        if p.exists() and 'Traceback' in p.read_text():raise RuntimeError('development failed; no sealed access')
        t=Path('reports/m5o_training.json')
        if t.exists() and json.loads(t.read_text())['status']!='Completed':
            print('Support insufficient; no sealed/model expansion',flush=True);return
        time.sleep(5)
    run('tools.m5o_model_inference_check',[],'inference_equivalence.log','.venv-gpu/bin/python')
    # Only one owner of simulator tick. Development campaign must have completed.
    run('tools.m5o_sealed_run',[],'sealed_run.log','.venv-gpu/bin/python')
    for split in ['train','val','calibration']:
        if not Path(f'reports/m5o_candidate_quality_{split}.json').exists():
            run('tools.m5o_candidate_audit',['--split',split],f'candidate_{split}.log','.venv-gpu/bin/python')
    for split in ['train','val','calibration','sealed']:
        if not Path(f'reports/m5o_all_candidate_burden_{split}.json').exists():
            run('tools.m5o_all_candidate_burden',['--split',split],f'burden_{split}.log','.venv-gpu/bin/python')
        if not Path(f'reports/m5o_capture_checks_{split}.json').exists():
            run('tools.m5o_dataset_checks',['--split',split],f'capture_checks_{split}.log','.venv-gpu/bin/python')
    run('tools.m5o_sampling_audit',[],'sampling_diversity.log','.venv-gpu/bin/python')
    print('Experiments completed; final interpretation and historical integrity audit remain',flush=True)
if __name__=='__main__':main()
