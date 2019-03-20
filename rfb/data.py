import numpy as np
import os

def _remove_missing(path):
    clean = path+'-known'
    if not os.path.isfile(clean+".data"):
        with open(clean+".data",'w') as oc, open(path+".data") as f:
            for l in f:
                if "?" not in l:
                    oc.write(l)
    return clean

def _letter(path, l1, l2):
    X = np.genfromtxt(path+"letter/letter-recognition.data",delimiter=",",dtype=str)
    y = X[:,0]
    X = X[:,1:]
    idx = (y == l1) | (y == l2)
    return X[idx],y[idx]
def _ilpd(path):
    X=np.genfromtxt(path+"ilpd/ilpd.csv",delimiter=",",dtype=str)
    y = X[:,-1]
    X = X[:,:-1]
    return X,y
def _mushroom(path):
    X=np.genfromtxt(path+"mushrooms/mushrooms.data",delimiter=",",dtype=str)
    y = X[:,0]
    X = X[:,1:]
    return X,y
def _tictactoe(path):
    X = np.genfromtxt(path+"tic-tac-toe/tic-tac-toe.data",delimiter=",",dtype=str)
    y = X[:,-1]
    X = X[:,:-1]
    return X,y 
def _sonar(path):
    X = np.genfromtxt(path+"sonar/sonar.data",delimiter=",",dtype=str)
    y = X[:,-1]
    X = X[:,:-1]
    return X,y 
def _usvotes(path):
    data  = path+"usvotes/house-votes-84"
    clean = _remove_missing(data)
    X = np.genfromtxt(clean+'.data',delimiter=",",dtype=str)
    y = X[:,0]
    X = X[:,1:]
    return X,y 
def _wdbc(path):
    X = np.genfromtxt(path+"wdbc/wdbc.data",delimiter=",",dtype=str)
    y = X[:,1]
    X = X[:,2:]
    return X,y 
def _heart(path):
    data  = path+"heart/cleveland"
    clean = _remove_missing(data)
    X = np.genfromtxt(clean+".data",delimiter=",",dtype=str)
    y = X[:,-1].astype(int)
    # Make binary
    y[y>0] = 1
    X = X[:,:-1]
    return X,y 
def _haberman(path):
    X = np.genfromtxt(path+'haberman/haberman.data',delimiter=",",dtype=str)
    y = X[:,-1]
    X = X[:,:-1]
    return X,y 
def _ionosphere(path):
    X = np.genfromtxt(path+'ionosphere/ionosphere.data',delimiter=",",dtype=str)
    y = X[:,-1]
    X = X[:,:-1]
    return X,y 
def _credita(path):
    data  = path+"credit-a/crx"
    clean = _remove_missing(data)
    X=np.genfromtxt(clean+".data",delimiter=",",dtype=str)
    y = X[:,-1]
    X = X[:,:-1]
    return X,y

DATA_SETS = {
        'Letter:AB':   lambda p: _letter(p, 'A', 'B'),
        'Letter:DO':   lambda p: _letter(p, 'D', 'O'),
        'Letter:OQ':   lambda p: _letter(p, 'O', 'Q'),
        'Sonar':       _sonar,
        'USVotes':     _usvotes,
        'WDBC':        _wdbc,
        'Heart':       _heart,
        'Haberman':    _haberman,
        'Ionosphere':  _ionosphere,
        'ILPD':        _ilpd,
        'Mushroom':    _mushroom,
        'Tic-Tac-Toe': _tictactoe,
        'Credit-A':    _credita
        }

def _relabel(V):
    assert(len(V.shape) == 2)
    for i in range(0, V.shape[1]):
        try:
            float(V[0,i])
        except ValueError:
            mp = dict()
            c  = 0
            for j in range(V.shape[0]):
                if V[j,i] not in mp:
                    mp[V[j,i]] = c
                    c += 1
                V[j,i] = mp[V[j,i]]
    return V

def split(X, Y, f):
    n = X.shape[0]
    s = int(float(f)*float(n))
    return X[:s],Y[:s],X[s:],Y[s:]


def load(dataset, path='data/', seed=123):
    assert(dataset in DATA_SETS)

    np.random.seed(seed)

    X,Y = DATA_SETS[dataset](path)
    X = _relabel(X)
    Y = np.reshape(_relabel(np.reshape(Y,(Y.shape[0],1))), (-1,))

    X = X.astype(float)
    Y = Y.astype(int)

    perm = np.random.permutation(X.shape[0])
    X,Y = X[perm], Y[perm]
    
    return X,Y
