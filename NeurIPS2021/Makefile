experiments : clean data link

data: 
	./helpers/get_data.sh

link:
	ln -s ../mvb mvb

uniform:
	./helpers/run_uniform.sh bootstrap

optimize:
	./helpers/run_optimize.sh bootstrap

mu:
	./helpers/run_mu.sh bootstrap

clean:
	rm -f mvb
	rm -rf data
	rm -rf out
