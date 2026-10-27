"""Summarize the completed retry; read-only toward models, data, and old reports.

No model import, GPU operation, or optimizer update. Refuses overwriting reports.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_files(rows):
    for row in rows:
        assert sha(row['path']) == row['sha256'], row['path']
        if 'bytes' in row:
            assert Path(row['path']).stat().st_size == row['bytes'], row['path']
    return dict(files=len(rows), unchanged=True)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    protocol = read('configs/m8j6_retry_protocol.json')
    run = protocol['run_id']
    root = Path('reports/m8j6_retry_sources') / run
    supervisor = read(root / 'supervisor.json')
    phases = [read(root / p / 'result.json') for p in ('first', 'resume', 'verify')]
    mapping = read(root / 'first/parameter_mapping.json')
    assert supervisor['status'] == 'Passed'
    assert [p['optimizer_calls_this_process'] for p in phases] == [2, 1, 0]
    assert all(p['status'] == 'PhasePassed' for p in phases)
    assert len({p['process_pid'] for p in phases}) == 3
    steps = [read(root / phase / ('step_%d.json' % step))
             for phase, step in [('first', 1), ('first', 2), ('resume', 3)]]
    ledgers = sorted(root.glob('*/optimizer_step_*_returned.json'))
    assert sorted(read(p)['global_step'] for p in ledgers) == [1, 2, 3]
    assert [s['frame_index'] for s in steps] == [0, 4, 1]
    assert steps[0]['optimizer_before']['entries'] == 0
    assert steps[0]['before_state_sha256'] == phases[0]['original_state_sha256']
    for i, s in enumerate(steps):
        assert s['gradient_driven_changed_parameters'] > 0
        assert s['frozen_parameter_and_buffer_max_abs_change'] == 0
        assert s['all_buffers_max_abs_change'] == 0
        assert s['gradients']['all_finite']
        assert s['optimizer_after']['weight_decay'] == [0.0]
        assert s['optimizer_after']['step_values'] == [i + 1]
        assert len(s['layers']) == 6
        assert all(l['P'] + l['I'] + l['N'] == 900 for l in s['layers'])
        if i:
            assert s['before_state_sha256'] == steps[i - 1]['after_state_sha256']
            assert s['optimizer_before'] == steps[i - 1]['optimizer_after']
    for p in ('resume', 'verify'):
        assert read(root / p / 'parameter_mapping.json') == mapping
    restores = [read(root / p / 'restore.json') for p in ('resume', 'verify')]
    for r in restores:
        assert r['passed'] and r['new_process']
        for key in ('model_parameters_and_buffers_exact', 'optimizer_exact',
                    'parameter_sharing_preserved', 'optimizer_mapping_exact',
                    'freeze_policy_exact', 'RNG_exact', 'module_modes_exact'):
            assert r[key]
        assert all(v['passed'] for v in r['fixed_eval_output_comparison'].values())

    preflight = read('reports/m8j6_retry_preflight.json')
    prepared = Path('/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/prepared')
    manifest = read(prepared / 'manifest.json')
    assert sha(prepared / 'manifest.json') == preflight['prepared_manifest_sha256']
    data_files = []
    for frame in manifest['frames']:
        for kind, field in [('rgb_input', 'input_sha256'), ('supervision', 'supervision_sha256')]:
            data_files.append(dict(path=str(prepared / kind / ('%02d.npz' % frame['index'])),
                                   sha256=frame[field]))
    old_changes = subprocess.check_output(
        ['git', 'diff', '--name-status', '--diff-filter=MD', protocol['base_commit']], text=True)
    assert not old_changes, old_changes
    preservation = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        old_release=check_files(read('reports/m8j6_actual_release_manifest.json')['files']),
        original_model_and_source=check_files(read('reports/m8j6_source_lock.json')['assets']),
        old_private_attempt=check_files(preflight['old_attempt']['private_files']),
        eight_fixed_input_and_supervision_bundles=check_files(data_files),
        prepared_manifest_unchanged=True,
        execution_sources=check_files([dict(path=p, sha256=h)
            for p, h in supervisor['source_sha256'].items()]),
        execution_logs=check_files([dict(path=str(root / p['phase'] / 'execution.txt'),
            sha256=p['log_sha256']) for p in supervisor['phases']]),
        new_private_checkpoints=check_files([p['checkpoint'] for p in phases[:2]]),
        historical_tracked_files_modified_or_deleted=[],
        owned_pids=[dict(pid=p['process_pid'], still_present=Path('/proc/%d' % p['process_pid']).exists())
                   for p in phases],
        original_checkpoint_used=True, checkpoint_tensor_upload=False)
    assert not any(p['still_present'] for p in preservation['owned_pids'])

    updates = dict(run_id=run, evidence_type='ActualRTX4090_FP32_RealNuScenesMini',
        code_commit=supervisor['code_commit'], old_attempt_optimizer_steps=1,
        new_attempt_optimizer_steps=3, total_across_separate_attempts=4,
        fresh_optimizer_initial_entries=0, sample_order=[0, 4, 1],
        original_weight_sha256=phases[0]['initial_weight_sha256'],
        initial_safe_archive_sha256=phases[0]['safe_archive_sha256'],
        parameter_mapping_path=str(root / 'first/parameter_mapping.json'),
        unique_parameters=len(mapping['parameters']),
        unique_trainable_parameters=mapping['canonical_trainable_count'],
        trainable_state_aliases=len(mapping['trainable_state_names']),
        frozen_state_tensors=len(mapping['frozen_state_names']),
        frozen_unique_parameters=sum(not p['trainable'] for p in mapping['parameters']),
        buffers=len(mapping['buffer_names']), optimizer_duplicates=0, alias_policy_conflicts=0,
        trainable_numel=phases[0]['trainable_numel'], frozen_numel=phases[0]['frozen_numel'],
        parameter_gradient_norm_scope='Per-parameter and module norms are before clipping; global norms report before and after clipping.',
        steps=[{k:v for k,v in s.items() if k not in ('parameter_gradient_norms', 'trainable_parameter_changes')}
               for s in steps],
        detailed_changes=[str(root / phase / ('step_%d.json' % step))
                          for phase,step in [('first',1),('first',2),('resume',3)]])
    acceptance = dict(stage='M8-J6 Independent Three-Step FP32 Retry',
        checked_at_utc=preservation['checked_at_utc'], base_commit=protocol['base_commit'],
        code_commit=supervisor['code_commit'], run_id=run, status='SmokePassed',
        checker_regression=dict(status='Passed', reused_tests=36, new_tests=14, total=50,
            evidence=['reports/m8j6_retry_sources/reused20.txt',
                      'reports/m8j6_retry_sources/reused16',
                      'reports/m8j6_retry_sources/new_state_tests_final.txt']),
        old_attempt=dict(status='FailedStopped_Preserved', actual_optimizer_steps=1,
            checkpoint='NotSaved_HistoricalEvidenceGap',
            parameter_difference='NotPersisted_HistoricalEvidenceGap',
            source='reports/m8j6_sources/actual_smoke/result.json'),
        new_attempt=dict(status='Passed', actual_optimizer_steps=3, process_steps=[2,1,0],
            fresh_baseline=True, fresh_optimizer=True, fixed_sample_order=[0,4,1]),
        frozen_parameters_buffers='Passed_ExactUnchanged',
        shared_parameters_and_optimizer_mapping='Passed',
        step2_cross_process_restore='Passed', resumed_step3='Passed',
        step3_independent_reload='Passed',
        fixed_eval_max_abs_error=max(v['max_abs_error'] for r in restores
                                    for v in r['fixed_eval_output_comparison'].values()),
        peak_step_allocated_bytes=max(s['peak_allocated_bytes'] for s in steps),
        peak_step_reserved_bytes=max(s['peak_reserved_bytes'] for s in steps),
        memory_scope='Per-step PyTorch CUDA allocator; head-only backward, excludes driver and other processes.',
        FP32='Passed_ThisFixedSmokeOnly', AMP='NotRetested_NotPassed',
        overfit='NotRun_SeparateBoundedExperimentRecommended',
        generalization='NotEvaluated', visual_driving_safety='NotValidated',
        all_uninterrupted_training_bitwise_equivalence='NotClaimed',
        historical_assets='Unchanged', private_assets_published=False,
        CARLA='NotRun', main_merge=False)
    write('reports/m8j6_retry_preservation.json', preservation)
    write('reports/m8j6_retry_updates.json', updates)
    write('reports/m8j6_retry_restore_validation.json', dict(run_id=run,
        code_commit=supervisor['code_commit'], restores=restores,
        checkpoints=[{k:v for k,v in p['checkpoint'].items() if k != 'saved_model_modes'} for p in phases[:2]],
        third_process_optimizer_steps=0))
    write('reports/m8j6_retry_acceptance.json', acceptance)
    print(json.dumps(dict(status=acceptance['status'], steps=3, preserved=True)))


if __name__ == '__main__':
    main()
