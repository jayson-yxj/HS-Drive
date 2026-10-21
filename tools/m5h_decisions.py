"""Record optional research decisions based on validation only."""
import json
from pathlib import Path
from simulator.recorder import atomic_json,file_hash

def main():
    p=Path('reports/m5h_validation_decisions.json')
    if p.exists():raise FileExistsError(p)
    pix=json.loads(Path('reports/m5h_distillation_ablation.json').read_text());dep=json.loads(Path('reports/m5h_depth_preservation.json').read_text())
    atomic_json(p,dict(kind='validation_only_research_decision',hardnegative=dict(eligible=pix['conditional_hard_eligible'],status='NotRun',reason='Noautomaticextension; conservative KD cannotclaimstableimprovement unlessbothseeds pass'),backbone=dict(status='NotRun',reason='Addeddata still only2positive rear_right views; exposurecounts do notruleout appearancecoverage'),depth_D=dict(eligible=dep['spatial_D_condition'],status='Deferred',reason='OptionalD isallowed notmandatory; matchedB/C alreadyprovide nearpreservingcandidates. Freeze andtest these first; avoidconfoundingspatialloss with retentionresearch.'),source_sha256=file_hash(__file__)))
if __name__=='__main__':main()
