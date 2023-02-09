M=100
R=20

for F in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
do
	python unlabeled.py Mushroom $M $F $R
	python unlabeled.py Phishing $M $F $R
	python unlabeled.py Adult    $M $F $R
	python unlabeled.py w1a      $M $F $R
	python unlabeled.py Cod-RNA  $M $F $R
done
