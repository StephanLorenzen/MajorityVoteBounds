.PHONY: sparsification

sparsification:
	cd sparsification; pdflatex sparsify_Pendigits.tex; rm -f sparsify_Pendigits.{log,aux}
	cd sparsification; pdflatex sparsify_Cod-RNA.tex; rm -f sparsify_CoD-RNA.{log,aux}
	cd sparsification; pdflatex sparsify_MNIST.tex; rm -f sparsify_MNIST.{log,aux}
	cd sparsification; pdflatex sparsify_Adult.tex; rm -f sparsify_Adult.{log,aux}
	cd sparsification; pdflatex sparsify_Shuttle.tex; rm -f sparsify_Shuttle.{log,aux}
	cd sparsification; pdflatex sparsify_Sensorless.tex; rm -f sparsify_Sensorless.{log,aux}
	cd sparsification; pdflatex sparsify_Phishing.tex; rm -f sparsify_Phising.{log,aux}
	rm -f texput.log


clean:
	cd sparsification; rm -f *.pdf;
