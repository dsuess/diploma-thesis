IMGNAMES = scaling.pdf
IMGPATHS = $(addprefix img/, $(IMGNAMES))

TIKZFILES = cubic_hierarchy triang_hierarchy
TIKZPATHS = $(addprefix img/, $(addsuffix .tikz, $(TIKZFILES)))
TIKZPDF = $(addprefix img/, $(addsuffix .pdf, $(TIKZFILES)))

$(info $(TIKZPATHS))

thesis: thesis.tex $(IMGPATHS) $(TIKZPATHS)
	latexmk -pdf -pv -f thesis.tex

$(IMGPATHS): img/numerics.py
	cd img; ./numerics.py

$(TIKZPDF): $(TIKZPATHS)
	for file in $(TIKZFILES) ; do\
		cd img/; pdflatex -interaction=nonstopmode $(addsuffix .tikz, $$file) ; \
	done

clean:
	cd img/; rm -f *.aux *.log
	rm -f *.log *.aux *.blg *.fls *.fdb_latexmk *.out *.log *.bbl *.auxlock
