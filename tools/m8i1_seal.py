"""Seal the completed failure review; never launches CARLA or overwrites assets."""
import json
import subprocess
from pathlib import Path
from tools.m8i1_common import ROOT, BASE, asset, write

MANIFEST = Path('reports/m8i1_release_manifest.json')
RECEIPT = Path('reports/m8i1_logs/post_seal_verification.json')


def main():
    assert not MANIFEST.exists(), 'Existing seal must not be replaced'
    result = json.loads((ROOT / 'session_result.json').read_text())
    files = set()
    for directory in ['tools', 'datasets', 'simulator', 'tests', 'reports', 'docs', 'configs/m8i1']:
        for path in Path(directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and (
                any('m8i1' in part.lower() for part in path.parts)
            ) and path not in (MANIFEST, RECEIPT):
                files.add(path)
    files.update(p for p in ROOT.rglob('*') if p.is_file())
    write(MANIFEST, dict(
        kind='OfflineReleaseSeal_NoRuntimeExecution', base=BASE,
        executed_commit=result['execution_commit'],
        report_head_before_final_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        files=[asset(p) for p in sorted(files)],
        self_excluded=True, post_seal_receipt_excluded=str(RECEIPT),
        provenance_note='CARLA execution sources verified against executed_commit; '
            'offline working-tree tools are pinned by this manifest, not the pre-final HEAD. '
            'Post-shutdown source directory snapshot includes an unexecuted, removed evaluator '
            'draft; its hash is not proof of execution and source is unavailable.',
        model_forwards=0, training_steps=0, visual_control=False,
        split='DevelopmentOnly', FinalEligible=False,
    ))
    print(f'Sealed {len(files)} files; no simulator connection')


if __name__ == '__main__':
    main()
