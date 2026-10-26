"""Bounded archive validation and extraction into a new private directory only."""
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from collections import Counter
from tools.m8j3_download_clip import PRIVATE, NAME, SIZE, SHA, now

MAX_MEMBERS = 100000
MAX_BYTES = 8 * 1024**3
MAX_FILE = 128 * 1024**2


def validate_member(member, seen):
    name = member.name
    p = PurePosixPath(name)
    if (not name or '\\' in name or '\x00' in name or p.is_absolute()
            or '..' in p.parts or ':' in name or any(ord(c) < 32 for c in name)):
        raise ValueError('Unsafe path')
    normalized = str(p)
    if normalized in ('', '.') or normalized in seen:
        raise ValueError('Duplicate/empty normalized path')
    if not (member.isdir() or member.isreg()) or member.issparse():
        raise ValueError('Links, devices, sparse or unsupported entries are forbidden')
    if member.size < 0 or member.size > MAX_FILE:
        raise ValueError('Single-file budget exceeded')
    seen.add(normalized)
    return normalized


def main():
    os.umask(0o077)
    src = PRIVATE / 'download' / NAME
    h = hashlib.sha256()
    with src.open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    assert src.stat().st_size == SIZE and h.hexdigest() == SHA
    expanded = 0
    with gzip.open(src, 'rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            expanded += len(block)
            if expanded > MAX_BYTES:
                raise ValueError('Gzip expanded byte budget exceeded')
    seen, members, total, types = set(), [], 0, Counter()
    with tarfile.open(src, 'r:gz') as tar:
        for m in tar:
            path = validate_member(m, seen)
            total += m.size
            if len(members) >= MAX_MEMBERS or total > MAX_BYTES:
                raise ValueError('Archive budget exceeded')
            members.append(dict(path=path, bytes=m.size, directory=m.isdir()))
            types['directory' if m.isdir() else PurePosixPath(path).suffix] += 1
    dest = PRIVATE / 'extracted'
    dest.mkdir(mode=0o700, exist_ok=False)
    with tarfile.open(src, 'r:gz') as tar:
        for m in tar:
            p = dest / str(PurePosixPath(m.name))
            if not p.resolve().is_relative_to(dest.resolve()):
                raise ValueError('Extraction escaped destination')
            if m.isdir():
                p.mkdir(parents=True, exist_ok=True, mode=0o700)
                continue
            p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            reader = tar.extractfile(m)
            count = 0
            with p.open('xb') as out:
                for block in iter(lambda: reader.read(1024**2), b''):
                    count += len(block)
                    if count > m.size:
                        raise ValueError('Member size overrun')
                    out.write(block)
            if count != m.size:
                raise ValueError('Truncated member')
    with (PRIVATE / 'archive_members.json').open('x') as f:
        json.dump(members, f, indent=2)
    result = dict(status='GzipCRCAndTarSafetyPassed_Extracted', checked_at_utc=now(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_archive_sha256=SHA, members=len(members), regular_bytes=total,
        gzip_expanded_bytes=expanded, types=dict(types), max_members=MAX_MEMBERS,
        max_expanded_bytes=MAX_BYTES, max_single_file_bytes=MAX_FILE,
        forbidden_types_found=0, duplicate_paths_found=0, extracted_to=str(dest),
        executed_archive_code=False, symlinks_created=False, tar_extractall_used=False,
        owner_modes_restored=False, scope='LocalPrivateOnly')
    with (PRIVATE / 'archive_result.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
