SHELL=/bin/bash
.PHONY: rho

optimize_rfc:
	python plot.py rfc 100
	cd figure/rfc; pdflatex selected.tex
	cd figure/rfc; pdflatex multiclass.tex
	cd figure/rfc; pdflatex binary.tex
	cd figure/rfc; rm -f *.{log,aux}; rm -f texput.log
	rm -f texput.log

optimize_mce:
	python plot.py mce 7
	cd figure/mce; pdflatex selected.tex
	cd figure/mce; pdflatex multiclass.tex
	cd figure/mce; pdflatex binary.tex
	cd figure/mce; rm -f *.{log,aux}; rm -f texput.log
	rm -f texput.log

clean:
	cd figure/mce; rm -f *.pdf; rm -rf datasets
	cd figure/rfc; rm -f *.pdf; rm -rf datasets
