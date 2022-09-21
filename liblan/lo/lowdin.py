import numpy as np
import scipy.linalg

def lowdin(s):
    ''' new basis is |mu> c^{lowdin}_{mu i} '''
    e, v = scipy.linalg.eigh(s)
    idx = e > 1e-15
    return np.dot(v[:,idx]/np.sqrt(e[idx]), v[:,idx].conj().T)

def caolo(s):
    return lowdin(s)

def cloao(s):
    return lowdin(s) @ s 