# MCSCF solvers for DMET impurity 

from pyscf import gto, scf, mcscf, fci
from pyscf.lib import logger

import h5py
import numpy as np

def sacasscf_solve_imp(solver_base,mol,ncasorb,ncaselec,casorbind,statelis,avas=False):
    solver = mcscf.CASSCF(solver_base,ncasorb,ncaselec)

    if avas:
        mo = casorbind
    else:
        mo = solver.sort_mo(casorbind)  
    solvers = []
    logger.info(solver,'Attempting SA-CASSCF with')
    for i in range(len(statelis)):
        if i == 0 and statelis[0]:
            newsolver = fci.direct_spin0.FCI(mol)
            newsolver.spin = 0
            newsolver.nroots = statelis[0]
            solvers.append(newsolver)
            logger.info(solver,'%s states with spin multiplicity %s',statelis[0],0+1)
        elif statelis[i]:
            newsolver = fci.direct_spin1.FCI(mol)
            newsolver.spin = i
            newsolver = fci.addons.fix_spin(newsolver,ss=(i/2)*(i/2+1),shift=.1)
            newsolver.nroots = statelis[i]
            solvers.append(newsolver)
            logger.info(solver,'%s states with spin multiplicity %s',statelis[i],i+1)

    statetot = sum(statelis)
    mcscf.state_average_mix_(solver, solvers, np.ones(statetot)/statetot)
    solver.run(mo,verbose = mol.verbose)

    return solver

def sacasscf_dump_chk(solver,sav):
    # saving properties for SISO
    with h5py.File(sav, 'w') as fh5:
        fh5['ncore'] = np.asarray(solver.ncore)
        fh5['ncas'] = np.asarray(solver.ncas)
        fh5['nelecas'] = np.asarray(solver.nelecas)
        fh5['mo_coeff'] = solver.mo_coeff
        fh5['rdm1'] = solver.make_rdm1()

        for i in range(len(solver.ci)):
            fh5['ci'+str(i)] = solver.ci[i]
        fh5['e_states'] = solver.e_states

def sacasscf_load_chk(solver,sav):
    with h5py.File(sav, 'r') as fh5:
        solver.ncore = fh5['ncore'][()]
        solver.ncas = fh5['ncas'][()]
        solver.nelecas = fh5['nelecas'][:]

        solver.mo_coeff = fh5['mo_coeff'][:]
        rdm1 = fh5['rdm1'][:]
        solver.make_rdm1 = lambda *args: rdm1

        solver.e_states = fh5['e_states'][:]
        solver.ci = [fh5['ci'+str(i)][:] for i in range(len(solver.e_states))]
    return solver
