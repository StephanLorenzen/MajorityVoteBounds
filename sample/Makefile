example : data link

data: 
	mkdir data
	mkdir data/letter
	wget -O data/letter/letter-recognition.data https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data

link:
	ln -s ../mvb mvb

run:
	python example.py

clean:
	rm -f mvb
	rm -f data/letter/letter-recognition.data
	rm -df data/letter
	rm -df data
