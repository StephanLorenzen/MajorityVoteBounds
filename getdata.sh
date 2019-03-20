#!/bin/sh

mkdir data
cd data

# Get letter
mkdir letter
wget -O letter/letter-recognition.data https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data

# Get ilpd
mkdir ilpd
wget -O ilpd/ilpd.csv 'https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv'

# Get mushroom
mkdir mushroom
wget -O mushroom/mushroom.data https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data

# Get tic-tac-toe
mkdir tic-tac-toe
wget -O tic-tac-toe/tic-tac-toe.data https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data

# Get sonar
mkdir sonar
wget -O sonar/sonar.data https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data

# Get usvotes 
mkdir usvotes
wget -O usvotes/house-votes-84.data https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data

# Get wdbc
mkdir wdbc
wget -O wdbc/wdbc.data https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

# Get heart
mkdir heart
wget -O heart/cleveland.data https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

# Get haberman
mkdir haberman
wget -O haberman/haberman.data https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data

# Get ionosphere
mkdir ionosphere
wget -O ionosphere/ionosphere.data https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data

# Get credit-a
mkdir credit-a
wget -O credit-a/crx.data https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data

