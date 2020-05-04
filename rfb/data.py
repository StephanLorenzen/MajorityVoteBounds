import numpy as np
import os

# Remove data points with missing entries
def _remove_missing(path, sfx='.data'):
    clean = path+'-known'
    if not os.path.isfile(clean+sfx):
        with open(clean+sfx,'w') as oc, open(path+sfx) as f:
            for l in f:
                if "?" not in l and ",," not in l:
                    oc.write(l)
    return clean

def _read_idx_file(path, d):
    X = []
    Y = []
    with open(path) as f:
        for l in f:
            x = np.zeros(d)
            l = l.strip().split()
            Y.append(int(l[0]))
            for pair in l[1:]:
                (i,v) = pair.split(":")
                x[int(i)-1] = float(v)
            X.append(x)
    return np.array(X),np.array(Y)

def _letter(path, l1, l2):
    X = np.genfromtxt(path+"letter/letter-recognition.data",delimiter=",",dtype=str)
    y = X[:,0]
    X = X[:,1:]
    if l1==None or l2==None:
        return X,y
    else:
        idx = (y == l1) | (y == l2)
        return X[idx],y[idx]
def _ilpd(path):
    data  = path+'ilpd/ilpd'
    clean = _remove_missing(data, sfx='.csv')
    X = np.genfromtxt(clean+'.csv',delimiter=",",dtype=str)
    y = X[:,-1]
    X = X[:,:-1]
    return X,y
def _mushroom(path):
    X=np.genfromtxt(path+"mushroom/mushroom.data",delimiter=",",dtype=str)
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
def _madelon(path):
    return _read_idx_file(path+"madelon/madelon.data", 500)
def _phishing(path):
    return _read_idx_file(path+"phishing/phishing.data", 68)
def _svmguide1(path):
    return _read_idx_file(path+"svmguide1/svmguide1.data", 4)
def _splice(path):
    return _read_idx_file(path+"splice/splice.data", 60)
def _adult(path):
    return _read_idx_file(path+"adult/adult.data", 123)
def _numer(path):
    return _read_idx_file(path+"numer/numer.data", 24)
def _w1a(path):
    return _read_idx_file(path+"w1a/w1a.data", 300)
def _mnist(path):
    return _read_idx_file(path+"mnist/mnist.data", 780)
def _shuttle(path):
    return _read_idx_file(path+"shuttle/shuttle.data", 9)
def _segment(path):
    return _read_idx_file(path+"segment/segment.data", 19)
def _pendigits(path):
    return _read_idx_file(path+"pendigits/pendigits.data", 16)

DATA_SETS = {
        'Letter':      lambda p: _letter(p, None, None),
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
        'Credit-A':    _credita,
        'Madelon':     _madelon,
        'Phishing':    _phishing,
        'SVMGuide1':   _svmguide1,
        'Splice':      _splice,
        'Adult':       _adult,
        'GermanNumer': _numer,
        'w1a':         _w1a,
        'mnist':       _mnist,
        'Shuttle':     _shuttle,
        'Segment':     _segment,
        'Pendigits':   _pendigits
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


def load(dataset, path='data/', seed=0):
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
