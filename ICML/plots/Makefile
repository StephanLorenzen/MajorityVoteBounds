uniform_bounds:
	python plot.py
	cd bounds_uniform; pdflatex bounds_uniform_bin.tex; rm -f bounds_uniform_bin.{log,aux}
	cd bounds_uniform; pdflatex bounds_uniform_mul.tex; rm -f bounds_uniform_mul.{log,aux}
	rm -f texput.log

clean:
	cd bounds_uniform; rm -f *.pdf; rm -rf datasets
