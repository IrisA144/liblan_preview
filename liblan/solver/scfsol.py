# MF solvers for DMET impurity 

from pyscf import gto, scf, ao2mo
from pyscf.lib import logger
from scipy import linalg

import numpy as np
import os


def rohf_solve_imp(hcore,eris,nelec,spin,max_mem,dm0=None,verbose=logger.INFO):
    mol = gto.M()
    mol.verbose = verbose
    mol.incore_anyway = True
    mol.nelectron = nelec
    mol.spin = spin

    nao = hcore.shape[0]
    print('nao',nao)

    mf = scf.rohf.ROHF(mol).x2c()
    mf.max_memory = max_mem
    mf.mo_energy = np.zeros((nao))
    mf.max_cycle = 1000
    mf.diis_space = 8
    mf.init_guess = '1e'
    mf.level_shift = .2

    mf.get_hcore = lambda *args: hcore
    mf.get_ovlp = lambda *args: np.eye(nao)
    mf._eri = ao2mo.restore(8, eris, nao)

    if dm0 is None:
        mf.kernel()
    else:
        if dm0.shape != hcore.shape:
            bath_size = hcore.shape[0]-dm0.shape[0]
            dm0 = linalg.block_diag(dm0,np.eye(bath_size)*(nelec-np.trace(dm0))/bath_size)
        mf.kernel(dm0)
    return mf


