experiments : clean data link

data: 
	./helpers/get_data.sh

link:
	ln -s ../mvb mvb

uniform:
	./helpers/run_uniform.sh bootstrap

optimize_rfc:
	./helpers/run_optimize.sh 100 rfc bootstrap

optimize_mce:
	./helpers/run_optimize.sh 7 mce bootstrap
	
optimize_abc:
	./helpers/run_optimize.sh 100 abc boost-2

mu:
	./helpers/run_mu.sh bootstrap
	
muBernstein:
	./helpers/run_muBernstein.sh bootstrap

clean:
	rm -f mvb
	rm -rf data
	rm -rf out
