"""Record actual completed pytest logs; never infer a pass from an exit-less log."""
import json,re,subprocess
from pathlib import Path
from simulator.recorder import atomic_json,file_hash

def main():
    out=Path('reports/m5i_test_results.json')
    if out.exists():raise FileExistsError(out)
    records=[]
    for p in [Path('reports/m5i_logs/full_tests_pretest.log'),Path('reports/m5i_logs/carla_python_tests.log')]:
        s=p.read_text();match=re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([\d.]+)s',s)
        assert match and 'FAILED' not in s and 'ERROR ' not in s,s[-2000:]
        records.append(dict(path=str(p),sha256=file_hash(p),passed=int(match[1]),skipped=int(match[2] or 0),elapsed_s=float(match[3])))
    gpu=json.loads(Path('reports/m5i_gpu_contract.json').read_text())
    atomic_json(out,dict(kind='actual_pytest_and_RTX4090_contract_execution',records=records,gpu_contract_sha256=file_hash('reports/m5i_gpu_contract.json'),passed=all(r['gt_mutation_identical']for r in gpu['records']),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=file_hash(__file__),limits='CARLA SDK environment unit tests overlap fullsuite; do notaddcounts. ActualCARLAcapture separatelyaudited, no driving safety claim.'))
if __name__=='__main__':main()
