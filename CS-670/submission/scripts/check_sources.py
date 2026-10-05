"""Check report dependencies, citations, data identities, and compilation logs.

Run after generating tables, and again after compilation. This reads saved
evidence only. Optional sibling-bundle comparisons are skipped in a standalone
source ZIP; bundled source hashes are always verified.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import shutil

ROOT=Path(__file__).resolve().parents[1]
QUALITY=ROOT/'quality'
QUALITY.mkdir(exist_ok=True)
bib=(ROOT/'references.bib').read_text()
keys=re.findall(r'@\w+\{([^,]+),',bib)
assert len(keys)==len(set(keys))==27
known=set(keys)
citations=set()
all_labels=set()
refs=set()
tex=list(ROOT.rglob('*.tex'))
for path in tex:
    if 'build' in path.relative_to(ROOT).parts:continue
    source=path.read_text()
    assert not any(ord(c)<32 and c not in '\t\r\n' for c in source),path
    for arg in re.findall(r'\\cite\w*\{([^}]+)\}',source):citations.update(arg.split(','))
    for arg in re.findall(r'\\(?:c|C)?ref\{([^}]+)\}',source):refs.update(arg.split(','))
    all_labels.update(re.findall(r'\\label\{([^}]+)\}',source))
    for name in re.findall(r'\\input\{([^}]+)\}',source):assert (ROOT/(name+'.tex')).exists(),name
    for name in re.findall(r'\\reportfigure\{([^}]+)\}',source):assert (ROOT/'figures'/name).exists(),name
assert citations<=known,citations-known
assert refs<=all_labels,refs-all_labels
prov=json.loads((QUALITY/'table_provenance.json').read_text())
for item in prov['inputs']:
    assert hashlib.sha256((ROOT/item['source']).read_bytes()).hexdigest()==item['sha256'],item
sibling=ROOT.parent
copied_data=list((ROOT/'data').rglob('*.csv'))
figs=list((ROOT/'figures').glob('*.pdf'))
assert len(figs)==9
archive_copies=False
if (sibling/'results/effective_noise/test_means.csv').exists():
    for p in copied_data:assert p.read_bytes()==(sibling/'results'/p.relative_to(ROOT/'data')).read_bytes(),p
    for p in figs:assert p.read_bytes()==(sibling/'figures'/p.name).read_bytes(),p
    archive_copies=True
pdfs={}
for stem in ['main','supplement']:
    log=ROOT/'build'/(stem+'.log')
    pdf=ROOT/'build'/(stem+'.pdf')
    if not pdf.exists():continue
    text=log.read_text(errors='replace') if log.exists() else ''
    issues=[l for l in text.splitlines() if ('undefined' in l.lower() and ('reference' in l.lower() or 'citation' in l.lower())) or 'multiply defined' in l.lower() or 'multiply-defined' in l.lower() or l.startswith('!')]
    assert not issues,(stem,issues)
    overfull=re.findall(r'Overfull \\[hv]box[^\n]*',text)
    pages=None
    if shutil.which('pdfinfo'):
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        match=re.search(r'^Pages:\s+(\d+)',info,re.M)
        if match:pages=int(match.group(1))
    pdfs[stem]={'pages':pages,'bytes':pdf.stat().st_size,'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
                'undefined_references_or_citations':0,'overfull_boxes':overfull}
report={'bibliography_entries':len(keys),'citation_keys_used':len(citations),
        'missing_citations':[],'missing_inputs':[],'unresolved_source_references':[],
        'vector_figures':len(figs),'csv_sources':len(copied_data),'original_copy_identities_checked':archive_copies,
        'narrative_source_words':sum(len(p.read_text().split()) for d in ['sections','appendices'] for p in (ROOT/d).glob('*.tex')),
        'compiled_documents':pdfs,'training_executed':False,'test_scoring_executed':False}
(QUALITY/'source_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
