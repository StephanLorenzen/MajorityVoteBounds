experiments : clean data link

data: 
	./helpers/get_data.sh

link:
	ln -s ../mvb mvb

uniform:
	./helpers/run_uniform.sh bootstrap
uniform-reduced:
	./helpers/run_uniform.sh 0.5
optimize:
	./helpers/run_optimize.sh bootstrap
optimize-reduced:
	./helpers/run_optimize.sh 0.5
unlabeled:
	./helpers/run_unlabeled.sh

clean:
	rm -f mvb
	rm -rf data
	rm -rf out
