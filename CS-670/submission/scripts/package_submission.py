"""Package the portable source and final PDFs; exclude build intermediates.

Run after source checks and compilation. No model or research archive is changed.
"""
from pathlib import Path
import hashlib
import zipfile

ROOT=Path(__file__).resolve().parents[1]
archive=ROOT/'CS670_LaTeX_Source.zip'
files=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(ROOT)
    if 'build' in rel.parts or '__pycache__' in rel.parts:continue
    if p==archive or p.name=='MANIFEST.sha256':continue
    files.append(p)
lines=[hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix() for p in files]
manifest=ROOT/'MANIFEST.sha256'
manifest.write_text('\n'.join(lines)+'\n')
files.append(manifest)
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in files:
        info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),date_time=(2026,10,5,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        info.external_attr=0o100644<<16
        z.writestr(info,p.read_bytes())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for line in z.read('MANIFEST.sha256').decode().splitlines():
        expected,path=line.split('  ',1)
        assert hashlib.sha256(z.read(path)).hexdigest()==expected,path
print(f'{archive.name}: {len(files)} files, {archive.stat().st_size:,} bytes; archive CRC and all internal hashes verified.')
