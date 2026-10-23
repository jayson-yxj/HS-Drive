"""Reproduce event findings in a fresh directory; never overwrite sealed reports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='New directory; must not exist')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.output:
        out = Path(args.output).resolve()
        out.mkdir(parents=True, exist_ok=False)
    else:
        out = Path(tempfile.mkdtemp(prefix='hsdrive-m6g-reproduce-'))
    (out / 'reports/m6g_execution_logs').mkdir(parents=True)
    for name in ['tools', 'diagnostics', 'configs']:
        (out / name).symlink_to(repo / name, target_is_directory=True)
    for name in ['m6f_trials_0916', 'm6f_execution_logs']:
        (out / 'reports' / name).symlink_to(repo / 'reports' / name, target_is_directory=True)
    env = dict(os.environ, PYTHONPATH=str(repo), GIT_DIR=subprocess.check_output(
        ['git', 'rev-parse', '--absolute-git-dir'], cwd=repo, text=True).strip())
    for module in ['m6g_analyze', 'm6g_details', 'm6g_figures']:
        with (out / 'reports/m6g_execution_logs' / (module + '.txt')).open('x') as log:
            subprocess.run([sys.executable, '-m', 'tools.' + module], cwd=out, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    keys = {
        'timeline_alignment': ['rows', 'trial_count', 'events', 'gear_events', 'delayed_request_rows', 'all25_preserved'],
        'braking_event_analysis': ['events', 'light_brake_counts', 'jerk_clusters', 'jerk_summary'],
        'startup_transmission_analysis': ['events', 'gear_events'],
        'physics_rebuild_analysis': ['events'],
        'runtime_overhead': ['RPC_ms', 'world_tick_call_ms', 'apply_interval_ms', 'trials', 'deadline_miss', 'skipped_wall_slots'],
    }
    checked = []
    for name, fields in keys.items():
        rel = Path('reports/m6g_' + name + '.json')
        actual = json.loads((out / rel).read_text())
        expected = json.loads((repo / rel).read_text())
        for key in fields:
            if actual[key] != expected[key]:
                raise RuntimeError('Reproduction differs: ' + name + '/' + key)
            checked.append(name + '/' + key)
    result = dict(kind='OfflineReproductionOfSealedM6FData', status='Passed', output=str(out),
                  exact_equal_fields=checked, new_CARLA_trials=0,
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                  figures='Regenerated; image bytes not compared because font/backend metadata can differ')
    (out / 'reproduction.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
