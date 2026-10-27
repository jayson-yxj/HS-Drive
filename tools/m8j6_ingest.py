"""Bounded Mini archive validation and safe private extraction; no downloads."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

from datasets.m8j6_selection import sha256


def member_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('UnsafeArchivePath:' + name)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() or args.report.exists():
        ap.error('Refuse overwrite')
    repo = Path(__file__).resolve().parents[1]
    if repo in args.output.resolve().parents:
        ap.error('Data must remain outside repository')
    start = time.monotonic()
    rec = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), archive=str(args.archive),
               origin='UserDownloadedOfficialMini_AfterBrowserLogin_PerCurrentConversation',
               original_URL='NotSupplied', official_expected_checksum=None,
               hash_claim='LocalReproducibility_NotOfficialAuthentication',
               compressed_bytes=args.archive.stat().st_size,
               sha256=sha256(args.archive), disk_free_bytes=shutil.disk_usage(args.output.parent).free)
    names = set(); entries = []; total = 0
    try:
        with gzip.open(args.archive, 'rb') as g:
            with tarfile.open(fileobj=g, mode='r|') as t:
                for m in t:
                    path = member_path(m.name)
                    if not (m.isfile() or m.isdir()):
                        raise ValueError('LinkOrSpecialMember:' + m.name)
                    if str(path) in names:
                        raise ValueError('DuplicateArchivePath:' + m.name)
                    names.add(str(path)); total += m.size
                    if len(names) > 150000 or total > 20 * 1024**3 or m.size > 1024**3:
                        raise ValueError('ArchiveQuotaExceeded')
                    entries.append(dict(path=str(path), bytes=m.size, directory=m.isdir()))
            while g.read(1024 * 1024):
                pass  # Validate complete gzip CRC/trailer, even beyond tar EOF.
        if rec['disk_free_bytes'] < total + 5 * 1024**3:
            raise ValueError('InsufficientDiskSpace')
        assert 'v1.0-mini/scene.json' in names and 'v1.0-mini/sample.json' in names
        rec.update(archive_integrity='GzipCRCAndTarStructurePassed', members=len(entries),
                   uncompressed_bytes=total, no_links_or_special_files=True, quota_bytes=20 * 1024**3)
        args.output.mkdir(mode=0o700)
        with tarfile.open(args.archive, 'r:gz') as t:
            for m in t:
                p = args.output / member_path(m.name)
                if m.isdir():
                    p.mkdir(mode=0o700, parents=True, exist_ok=True)
                else:
                    p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    with t.extractfile(m) as src, p.open('xb') as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                    if p.stat().st_size != m.size:
                        raise ValueError('ExtractedSizeMismatch:' + m.name)
                    p.chmod(0o600)
        with (args.output.parent / 'archive_inventory.json').open('x') as f:
            json.dump(entries, f)
        rec.update(status='ExtractedPrivateMini', data_root=str(args.output),
                   original_archive_modified=False, payload_files_in_git=False)
    except Exception as exc:
        rec.update(status='Failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        rec['elapsed_s'] = time.monotonic() - start
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as f:
            json.dump(rec, f, indent=2)
        print(json.dumps(rec, indent=2), flush=True)


if __name__ == '__main__':
    main()
