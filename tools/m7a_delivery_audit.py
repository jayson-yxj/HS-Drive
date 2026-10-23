"""Read-only delivery/reference audit, with an append-only result file."""
import json
import re
import subprocess
from pathlib import Path

from tools.m7a_common import asset, provenance, read, sha, write


REQUIRED = [
    'reports/m7a_summary.md', 'reports/m7a_acceptance.json',
    'reports/m7a_dynamics_research_closeout.md', 'reports/m7a_scenario_catalog.json',
    'reports/m7a_oracle_planning_benchmark.json',
    'reports/m7a_candidate_trajectory_quality.json',
    'reports/m7a_safety_semantics_audit.json',
    'reports/m7a_six_camera_shadow_contract.json',
    'reports/m7a_oracle_vision_isolation.json',
    'reports/m7a_bench2drive_feasibility.json', 'reports/m7a_runtime_status.json',
    'docs/M7A_IMPLEMENTATION.md', 'docs/M7A_ARCHITECTURE_DECISION.md',
]


def without_latency(value):
    if isinstance(value, dict):
        return {k: without_latency(v) for k, v in value.items()
                if k not in ('generation_ms', 'total_ms', 'latency_ms')}
    if isinstance(value, list):
        return [without_latency(v) for v in value
                if not (isinstance(v, list) and v and v[0] == 'latency_ms')]
    return value


def main():
    for p in REQUIRED:
        assert Path(p).is_file(), p
    reference_checks = {}
    json_files = sorted(Path('reports').glob('m7a*.json'))

    def check_refs(obj):
        if isinstance(obj, dict):
            if {'path', 'sha256'} <= obj.keys():
                path = obj['path']
                assert Path(path).is_file(), path
                assert sha(path) == obj['sha256'], ('reference SHA mismatch', path)
                reference_checks[path] = obj['sha256']
            for v in obj.values():
                check_refs(v)
        elif isinstance(obj, list):
            for v in obj:
                check_refs(v)

    for p in json_files:
        # Never accept JSON's nonstandard NaN / Infinity literals.
        obj = json.loads(p.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        check_refs(obj)

    markdown_checks = []
    for p in map(Path, (s for s in REQUIRED if s.endswith('.md'))):
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            link = (p.parent / target.split('#')[0]).resolve()
            # This audit output is created below after all assertions succeed.
            if link == Path('reports/m7a_delivery_validation.json').resolve():
                continue
            assert link.is_file(), (str(p), target)
            markdown_checks.append(dict(document=str(p), target=target))

    old = read('reports/m7a_planner_development_v2.json')
    new = read('reports/m7a_planner_development_reproduce.json')
    assert old['catalog'] == new['catalog'], 'catalog changed'
    assert without_latency(old['records']) == without_latency(new['records']), 'reproduction changed'
    base = read('reports/m7a_execution_logs/base_freeze.json')['base']
    changes = subprocess.check_output(['git', 'diff', '--name-status', base], text=True).splitlines()
    assert all(row.startswith('A\t') and 'm7a' in row.lower() for row in changes), changes
    # New execution did not alter any source used by the frozen evaluator.
    frozen_sources = ['diagnostics/m7a_planning.py', 'tools/m7a_scenarios.py',
                      'datasets/m7a_contracts.py', 'configs/m7a/protocol.json']
    for path in frozen_sources:
        assert subprocess.check_output(['git', 'show', '1eb7613:' + path]) == Path(path).read_bytes(), path

    write('reports/m7a_delivery_validation.json', dict(
        kind='ActualServerReadOnlyDeliveryAndDeterminismAudit', status='Passed',
        provenance=provenance(), required_files=[asset(p) for p in REQUIRED],
        strict_json_files=len(json_files), asset_references_checked=reference_checks,
        markdown_links_checked=markdown_checks, frozen_planner_sources_unchanged=frozen_sources,
        development_reproduction=dict(
            source=asset('reports/m7a_planner_development_v2.json'),
            repeat=asset('reports/m7a_planner_development_reproduce.json'),
            scenarios=28, catalog_exact=True, all_non_latency_record_values_exact=True,
            excluded_fields=['generation_ms', 'total_ms', 'latency_ms'],
            no_validation_or_final_reexecution=True),
        scope='Delivery/determinism only; does not upgrade safety, dynamics, perception or runtime gates',
        historical_bytes='Verified separately by m7a_release --seal'))
    print('Delivery audit passed:', len(REQUIRED), 'required files;', len(reference_checks), 'asset SHA references; development repeat exact')


if __name__ == '__main__':
    main()
