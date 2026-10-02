"""SHA-256 manifests that never hash themselves.

Written OUTSIDE the target directory, excluding every MANIFEST* file, then moved in. An existing manifest is kept
as MANIFEST.sha256.superseded_<UTC timestamp>, never overwritten.
"""

import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(target):
    target = Path(target)
    files = sorted(p for p in target.rglob("*") if p.is_file() and not p.name.startswith("MANIFEST"))
    lines = [f"{_sha(p)}  ./{p.relative_to(target).as_posix()}\n" for p in files]
    fd, tmp = tempfile.mkstemp(prefix="manifest_", suffix=".sha256")       # outside the target directory
    with os.fdopen(fd, "w") as f:
        f.writelines(lines)
    dest = target / "MANIFEST.sha256"
    if dest.exists():
        dest.rename(target / f"MANIFEST.sha256.superseded_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    shutil.move(tmp, dest)
    return len(lines)


def verify_manifest(target):
    """Return the list of relative paths whose hash does not match (empty = verified)."""
    target = Path(target)
    bad = []
    for line in (target / "MANIFEST.sha256").read_text().splitlines():
        h, rel = line.split("  ", 1)
        p = target / rel
        if not p.is_file() or _sha(p) != h:
            bad.append(rel)
    return bad
