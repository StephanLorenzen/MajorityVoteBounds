.PHONY: rho

mu:
	python plot.py
	cd mu_plot; pdflatex mu_bounds.tex; rm -f mu_bounds.{log,aux}
	cd mu_plot; pdflatex mu_stats.tex; rm -f mu_stats.{log,aux}
	rm -f texput.log

optimize_boost:
	python plot_optimize.py boost 300
	cd figure/boost; pdflatex mvrisk_optimized_comparison_AdaBs.tex
	cd figure/boost; pdflatex mvrisk_optimized_comparison_AdaPr.tex
	cd figure/boost; rm -f mvrisk_*.{log,aux}
	cd table/boost; pdflatex optimize_test_risk.tex
	cd table/boost; pdflatex optimize_bound.tex
	cd table/boost; rm -f optimize_*.{log,aux}
	#cd rho/boost; pdflatex plot_rhos.tex; rm -f plot_rhos.{log,aux}
	rm -f texput.log

optimize_rfc:
	python plot_optimize.py rfc 100
	cd figure/rfc; pdflatex mvrisk_optimized_comparison.tex
	cd figure/rfc; pdflatex bounds_optimized_comparison.tex
	cd figure/rfc; rm -f *_optimized_comparison.{log,aux}
	cd table/rfc; pdflatex optimize_test_risk.tex
	cd table/rfc; pdflatex optimize_bound.tex
	cd table/rfc; rm -f optimize_*.{log,aux}
	#cd rho/bootstrap; pdflatex plot_rhos.tex; rm -f plot_rhos.{log,aux}
	rm -f texput.log

optimize_mce:
	python plot_optimize.py mce 7
	cd figure/mce; pdflatex mvrisk_optimized_comparison.tex
	cd figure/mce; pdflatex bounds_optimized_comparison.tex
	cd figure/mce; rm -f *_optimized_comparison_mce.{log,aux}
	cd table/mce; pdflatex optimize_test_risk.tex
	cd table/mce; pdflatex optimize_bound.tex
	cd table/mce; rm -f optimize_*.{log,aux}
	rm -f texput.log

clean:
	cd figure/mce; rm -f *.pdf; rm -rf datasets
	cd figure/rfc; rm -f *.pdf; rm -rf datasets
	cd figure/boost; rm -f *.pdf; rm -rf datasets
	#cd mu_plot; rm -f *.pdf; rm -rf datasets
	#cd rho; rm -f *.pdf
	cd table/mce; rm -f *.pdf; rm -rf optimize
	cd table/rfc; rm -f *.pdf; rm -rf optimize
	cd table/boost; rm -f *.pdf; rm -rf optimize
