experiments : clean data link

data: 
	./helpers/get_data.sh

link:
	ln -s ../mvb mvb

uniform:
	./helpers/run_uniform.sh bootstrap

optimize:
	./helpers/run_optimize.sh boost 3

optimize_mce:
	./helpers/run_optimize_mce.sh bootstrap

mu:
	./helpers/run_mu.sh bootstrap
	
muBernstein:
	./helpers/run_muBernstein.sh bootstrap

clean:
	rm -f mvb
	rm -rf data
	rm -rf out
