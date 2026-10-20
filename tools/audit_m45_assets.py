"""Verify the frozen historical assets without altering sources or data."""
import argparse,json
from pathlib import Path
from simulator.recorder import file_hash,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',default='reports/m45_asset_freeze.json')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    frozen=json.loads(Path(args.freeze).read_text());differences={};counts={}
    for kind,values in frozen.items():
        counts[kind]=len(values);changed=[]
        for name,expected in values.items():
            path=Path(name)
            if not path.exists():changed.append([name,'missing']);continue
            if kind=='data_file_size_mtime_ns':
                stat=path.stat();actual=[stat.st_size,stat.st_mtime_ns]
            else:actual=file_hash(path)
            if actual!=expected:changed.append([name,actual])
        differences[kind]=changed
    passed=not any(differences.values())
    atomic_json(args.output,dict(passed=passed,counts=counts,differences=differences,
        freeze_sha256=file_hash(args.freeze),
        limitations='Original raw payloads checked by size+mtime; index/manifests and protected sources/configs/reports/checkpoints checked by SHA256, not all raw bytes.'))
    print(json.dumps(dict(passed=passed,counts=counts)))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
