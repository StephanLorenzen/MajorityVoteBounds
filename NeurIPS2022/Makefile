experiments : clean data link

data: 
	./helpers/get_data.sh

link:
	ln -s ../mvb mvb

optimize_rfc:
	./helpers/run_optimize.sh 100 rfc bootstrap

optimize_mce:
	./helpers/run_optimize.sh 7 mce bootstrap

clean:
	rm -f mvb
	rm -rf data
	rm -rf out
