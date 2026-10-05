"""Compile the two LaTeX entry points without touching experimental evidence.

Usage: python3 scripts/build_documents.py [--engine auto|latexmk|tectonic]
       [--tectonic-bin /path/to/tectonic]
The generated table fragments are already included; pandas is not needed here.
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import os

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--engine',choices=['auto','latexmk','tectonic'],default='auto')
parser.add_argument('--tectonic-bin',default='tectonic')
args=parser.parse_args()
engine=args.engine
if engine=='auto':
    engine='latexmk' if shutil.which('latexmk') else 'tectonic'
program='latexmk' if engine=='latexmk' else args.tectonic_bin
if not shutil.which(program):
    raise SystemExit('No selected LaTeX compiler is available. Use a TeX distribution or upload the source ZIP to Overleaf.')
(ROOT/'build').mkdir(exist_ok=True)
for src,dest in [('main.tex','CS670_Report.pdf'),('supplement.tex','CS670_Supplement.pdf')]:
    if engine=='latexmk':
        cmd=[program,'-pdf','-interaction=nonstopmode','-halt-on-error','-outdir=build',src]
    else:
        cmd=[program,'--keep-logs','--keep-intermediates','--untrusted','--outdir','build',src]
    subprocess.run(cmd,cwd=ROOT,check=True,env=os.environ.copy())
    shutil.copy2(ROOT/'build'/(Path(src).stem+'.pdf'),ROOT/dest)
    print('Created',dest)
