experiments : clean data link

data: 
	./helpers/get_data.sh

link:
	ln -s ../mvb mvb

optimize_rfc:
	./helpers/run_optimize.sh 100 rfc bootstrap

optimize_mce:
	./helpers/run_optimize.sh 7 mce bootstrap

artificial:
	cd ArtificialEvaluation; python PBBennett_vs_PBBernstein.py; python CCOracle_vs_TND.py 

clean:
	rm -f mvb
	rm -rf data
	rm -rf out
	rm -f artificialEvaluation/*.png
	rm -rf AdaBoost-test
