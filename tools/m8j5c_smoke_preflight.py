"""Read-only preflight for the concrete future Smoke proposal; never loads a model."""
import argparse
import json
from pathlib import Path


def preflight(config):
    reasons=[k for k,v in config.items() if k.startswith('currently_missing_') and v]
    if not config.get('execute_real_optimizer'):reasons.append('RealUpdatesNotAuthorized')
    return dict(status='NoGo' if reasons else 'NeedsIndependentExecutionReview',
        reasons=reasons,candidate_ordinals=config['candidate_ordinals'],
        precision=config['precision'],max_steps=config['max_optimizer_steps'],
        optimizer_steps=0,model_loaded=False,execution_implemented=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/m8j5c_smoke_proposal.json');args=ap.parse_args()
    result=preflight(json.loads(Path(args.config).read_text()))
    print(json.dumps(result,indent=2));raise SystemExit(2)


if __name__=='__main__':main()
