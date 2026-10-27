"""No GPU, no download. Missing source tables fail with an explicit result."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from datasets.m8j6_selection import official_splits, private_manifest, sha256

RUNTIME = Path('/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataroot', type=Path, required=True)
    ap.add_argument('--private-output', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, default=Path('configs/m8j6_protocol.json'))
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.private_output.resolve().is_relative_to(repo):
        ap.error('Private manifest must be outside repository')
    if args.report.exists() or args.private_output.exists():
        ap.error('Refuse to overwrite an existing result')
    split_path = RUNTIME / 'env/lib/python3.8/site-packages/nuscenes/utils/splits.py'
    protocol = json.loads(args.protocol.read_text())
    report = dict(stage='M8-J6', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=sha256(args.protocol), split_source_sha256=sha256(split_path),
                  model_loaded=False, optimizer_steps=0, payload_downloaded=False,
                  historical_evidence_modified=False, dataroot=str(args.dataroot.resolve()))
    try:
        manifest = private_manifest(args.dataroot, official_splits(split_path), protocol)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        report.update(status='Blocked_DataMissingOrInvalid', error_type=type(exc).__name__,
                      error=str(exc), real_frames_verified=0, smoke='NotRun')
        rc = 2
    else:
        args.private_output.parent.mkdir(parents=True, exist_ok=True)
        with args.private_output.open('x') as f:
            json.dump(manifest, f, indent=2, allow_nan=False)
        args.private_output.chmod(0o600)
        report.update(status='MetadataSelected_NotTrainingGo', real_frames_selected=len(manifest['records']),
                      private_manifest_sha256=sha256(args.private_output),
                      GT_geometry='NotRun', visual_review='NotRun', smoke='NotRun')
        rc = 0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
