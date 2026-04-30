# Makefile for LaTeX report
# Usage: make         (build PDF)
#        make clean   (remove build artifacts)

MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex custom.bib
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk \
	      $(MAIN).synctex.gz sections/*.aux

distclean: clean
	rm -f $(MAIN).pdf

.PHONY: all clean distclean
