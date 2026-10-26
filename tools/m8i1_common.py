"""M8-I1 append-only artifacts; no model imports or simulator side effects."""
import json
import subprocess
from pathlib import Path
from tools.m8a_common import sha, asset, write

BASE = '502f0701025f2de5a8f2cc1ecbdcd4b3cacd2b6a'
ROOT = Path('/root/gpufree-data/hs-drive-experiments/m8i1-traffic-development')
HISTORY = Path('/root/gpufree-data/hs-drive-experiments/m8g-placement-feasibility/unified_history_v1.json')
PROTOCOL = Path('configs/m8i1/protocol.json')


def provenance(kind):
    paths = [PROTOCOL, *Path('tools').glob('m8i1_*.py'),
             *Path('datasets').glob('m8i1_*.py'), *Path('simulator').glob('m8i1_*.py'),
             *Path('tests').glob('test_m8i1*.py')]
    return dict(kind=kind, base=BASE,
                execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_sha256={str(p):sha(p) for p in sorted(paths)},
                model_forwards=0, training_steps=0, visual_control=False,
                split='DevelopmentOnly', FinalEligible=False)


def log_json(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []
