"""Render completed diagnostic JSON without modifying previous research reports."""
import argparse
import json
from pathlib import Path
from simulator.recorder import atomic_json,file_hash
from diagnostics.report import plots,aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='reports/m4_depth_distribution_diagnostics.json')
    p.add_argument('--figures',default='reports/m4_depth_distribution_figures');p.add_argument('--summary',default='reports/m4_depth_distribution_bev_summary.json')
    a=p.parse_args()
    if Path(a.summary).exists():raise FileExistsError(a.summary)
    report=json.loads(Path(a.input).read_text())
    if report['status']!='complete' or not all(report['engineering_checks'].values()):raise ValueError('Incomplete diagnostics')
    summary=plots(report,Path(a.figures))
    atomic_json(a.summary,{'input_sha256':file_hash(a.input),'aggregation':'BEV continuous diagnostics are arithmetic frame means; road IoU pools confusion; same frame set for all models',
        'results':summary,'figures':{str(p):file_hash(p) for p in Path(a.figures).glob('*.png')}})
    print('Generated figures and BEV aggregation')

if __name__=='__main__':main()
