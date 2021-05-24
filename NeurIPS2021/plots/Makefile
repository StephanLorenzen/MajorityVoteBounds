.PHONY: rho

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

optimize_boost:
	python plot_optimize.py boost 300
	cd risk_comparison_optimized/boost; pdflatex mvrisk_optimized_comparison_AdaBs.tex; rm -f mvrisk_optimized_comparison_AdaBs.{log,aux}
	cd risk_comparison_optimized/boost; pdflatex mvrisk_optimized_comparison_AdaPr.tex; rm -f mvrisk_optimized_comparison_AdaPr.{log,aux}
	cd table/boost; pdflatex optimize_test_risk.tex; rm -f optimize_test_risk.{log,aux}
	cd table/boost; pdflatex optimize_bound.tex; rm -f optimize_bound.{log,aux}
	cd rho/boost; pdflatex plot_rhos.tex; rm -f plot_rhos.{log,aux}
	rm -f texput.log

optimize_rfc:
	python plot_optimize.py bootstrap 100
	cd risk_comparison_optimized/bootstrap; pdflatex mvrisk_optimized_comparison.tex; rm -f mvrisk_optimized_comparison.{log,aux}
	cd table/bootstrap; pdflatex optimize_test_risk.tex; rm -f optimize_test_risk.{log,aux}
	cd table/bootstrap; pdflatex optimize_bound.tex; rm -f optimize_bound.{log,aux}
	cd rho/bootstrap; pdflatex plot_rhos.tex; rm -f plot_rhos.{log,aux}
	rm -f texput.log


clean:
	cd bounds_uniform; rm -f *.pdf; rm -rf datasets
	cd mu_plot; rm -f *.pdf; rm -rf datasets
	cd rho; rm -f *.pdf
	cd risk_comparison_optimized; rm -f *.pdf; rm -rf datasets
	cd table; rm -f *.pdf; rm -rf optimize
