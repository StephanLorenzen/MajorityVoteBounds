uniform_bounds:
	python plot.py
	cd bounds_uniform; pdflatex bounds_uniform_bin.tex; rm -f bounds_uniform_bin.{log,aux}
	cd bounds_uniform; pdflatex bounds_uniform_mul.tex; rm -f bounds_uniform_mul.{log,aux}
	rm -f texput.log

optimize:
	python plot.py
	cd risk_comparison_optimized; pdflatex mvrisk_optimized_comparison.tex; rm -f mvrisk_optimized_comparison.{log,aux}
	rm -f texput.log


clean:
	cd bounds_uniform; rm -f *.pdf; rm -rf datasets
	cd mu_plot; rm -f *.pdf; rm -rf datasets
	cd risk_comparison_optimized; rm -f *.pdf; rm -rf datasets
