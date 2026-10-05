PYTHON ?= python3

.PHONY: all tables report supplement clean-help
all: report supplement

tables:
	$(PYTHON) scripts/generate_tables.py

report:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

supplement:
	latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

clean-help:
	@echo 'Generated auxiliaries may be removed with latexmk -c; frozen data and sources must be preserved.'
