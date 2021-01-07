uniform_bounds:
	python plot.py
	cd bounds_uniform; pdflatex bounds_uniform_bin.tex; rm -f bounds_uniform_bin.{log,aux}
	cd bounds_uniform; pdflatex bounds_uniform_mul.tex; rm -f bounds_uniform_mul.{log,aux}
	rm -f texput.log

mu:
	python plot.py
	cd mu_plot; pdflatex mu_bounds.tex; rm -f mu_bounds.{log,aux}
	cd mu_plot; pdflatex mu_stats.tex; rm -f mu_stats.{log,aux}
	rm -f texput.log

clean:
	cd bounds_uniform; rm -f *.pdf; rm -rf datasets
