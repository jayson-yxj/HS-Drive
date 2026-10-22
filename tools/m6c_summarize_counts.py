"""Final compact actual counts for the human report, no acceptance reinterpretation."""
import json
from pathlib import Path
from tools.m6c_common import write,provenance


def main():
    acceptance=json.loads(Path('reports/m6c_acceptance.json').read_text())
    allr=[json.loads(p.read_text()) for p in Path('reports/m6c_trials').glob('*/result.json')]
    notes=dict(**provenance('actual_completed_run_counts'),source='Privileged',
        count=len(allr),sim_ticks=sum(r['ticks'] for r in allr),
        all_physics_sha=sorted({r['physics']['sha256'] for r in allr}),
        all_control_over50=sum(r['over50'] for r in allr),all_control_over55=sum(r['over55'] for r in allr),
        all_control_over100=sum(r['over100'] for r in allr),all_deadline_miss=sum(r['deadline_miss'] for r in allr),
        all_skipped_slots=sum(r['skipped_wall_slots'] for r in allr),
        all_max_ms=max(r['control_interval_ms']['max'] for r in allr),
        none_inferred_safe_from_collision_count=True)
    write('reports/m6c_completed_counts.json',notes);print(json.dumps(notes,indent=2))

if __name__=='__main__':main()
