import numpy as np 
import os
import h5py

from scipy import linalg
from functools import reduce

from pyscf import lib, ao2mo, mcscf, fci
from pyscf.lib import logger
from pyscf.tools import molden

from liblan.lo import lowdin
from liblan.solver import scfsol, mcscfsol
from liblan.soc import siso

def read_cas_info(fname):
    with open(fname, 'r') as f1:
        ncasorb, ncaselec = [int(x) for x in f1.readline().split(' ')]
        casorbind = [int(x) for x in f1.readline().split(' ')]
    return ncasorb, ncaselec, casorbind

def read_sa_info(fname):
    with open(fname, 'r') as f1:
        statelis = [int(x) for x in f1.readline().split(' ')]
    return statelis

def get_dmet_as_props(mf,imp_inds,lo_meth='lowdin',thres=1e-13):
    """
    Returns C(AO->AS), entropy loss, and orbital composition
    """
    s = mf.get_ovlp()
    caolo, cloao = lowdin.caolo(s), lowdin.cloao(s)
    env_inds = [x for x in range(s.shape[0]) if x not in imp_inds]

    if mf.mol.spin == 0:
        pass
    else:
        dma, dmb = mf.make_rdm1()

        ldma = reduce(np.dot,(cloao,dma,cloao.conj().T))
        ldmb = reduce(np.dot,(cloao,dmb,cloao.conj().T))

        ldma_env = ldma[env_inds,:][:,env_inds]
        ldmb_env = ldmb[env_inds,:][:,env_inds]

        nat_occa, nat_coeffa = np.linalg.eigh(ldma_env)
        nat_occb, nat_coeffb = np.linalg.eigh(ldmb_env)

        ldmr_env = ldma_env + ldmb_env
        nat_occr, nat_coeffr = np.linalg.eigh(ldmr_env)
        
        caoas = np.hstack([caolo[:,imp_inds],caolo[:,env_inds] @ nat_coeffr])

        nuu = np.sum([nat_occr <  thres])
        ne = np.sum([(nat_occr >= thres) & (nat_occr <= 2-thres)])
        nuo = np.sum([nat_occr > 2-thres])

        asorbs = (nuu,ne,nuo)

        nat_occa = nat_occa[nat_occa > thres]
        nat_occb = nat_occb[nat_occb > thres]
        nat_occr = nat_occr[nat_occr > thres]

        ent = - np.sum(nat_occa*np.log(nat_occa)) - np.sum(nat_occb*np.log(nat_occb))
        entr = -2*np.sum(nat_occr/2*np.log(nat_occr/2))

        return caoas, entr - ent, asorbs

def get_dmet_imp_ldm(mf,imp_inds,lo_meth='lowdin'):
    """
    Returns better initial guess than '1e' for impurity
    """
    s = mf.get_ovlp()
    caolo, cloao = lowdin.caolo(s), lowdin.cloao(s)

    dma, dmb = mf.make_rdm1()

    ldm = reduce(np.dot,(cloao,(dma+dmb),cloao.conj().T))[imp_inds,:][:,imp_inds]
    return ldm

def get_as_1e_ints(mf,caoas,asorbs):
    # hcore from mf
    hcore = mf.get_hcore()

    # HF J/K from env UO
    uos = hcore.shape[0]-asorbs[-1]
    dm_uo = caoas[:,uos:] @ caoas[:,uos:].conj().T*2
    vj, vk = mf.get_jk(dm=dm_uo)

    fock = hcore + vj - 0.5 * vk 

    nimp = hcore.shape[0]-np.sum(asorbs)
    act_inds = [*list(range(nimp)),*list(range(nimp+asorbs[0],nimp+asorbs[0]+asorbs[1]))]

    asints1e = reduce(np.dot,(caoas[:,act_inds].conj().T,fock,caoas[:,act_inds]))
    return asints1e 

def get_as_2e_ints(mf,caoas,asorbs):
    nimp = caoas.shape[0]-np.sum(asorbs)
    act_inds = [*list(range(nimp)),*list(range(nimp+asorbs[0],nimp+asorbs[0]+asorbs[1]))]

    if mf._eri is not None:
        asints2e = ao2mo.full(mf._eri,caoas[:,act_inds])
    else:
        asints2e = ao2mo.full(mf.mol,caoas[:,act_inds])

    return asints2e 

def kernel(ssdmet,imp_inds,imp_solver,imp_solver_soc,statelis=None,thres=1e-13):
    """
    Driver function for SSDMET
    """

    mf = ssdmet.mf

    as_fname = ssdmet.title + '_as_chk.h5'
    if not os.path.isfile(as_fname):
        # AS general info
        caoas, ent, asorbs = get_dmet_as_props(mf,imp_inds,thres=thres)
        # AS hcore + j/k
        as1e = get_as_1e_ints(mf,caoas,asorbs)
        # AS ERIs
        as2e = get_as_2e_ints(mf,caoas,asorbs)
        with h5py.File(as_fname, 'w') as fh5:
            fh5['caoas'] = caoas
            fh5['ent'] = ent
            fh5['asorbs'] = asorbs
            fh5['1e'] = as1e 
            fh5['2e'] = as2e
    else:
        fh5 = h5py.File(as_fname, 'r')
        caoas = fh5['caoas'][:]
        ent = fh5['ent'][()]
        asorbs = fh5['asorbs'][:]
        as1e = fh5['1e'][:]
        as2e = fh5['2e'][:]
        fh5.close()

    logger.info(ssdmet,'Entanglement S: %.3f',ent)
    nimp = caoas.shape[0]-np.sum(asorbs)
    act_inds = [*list(range(nimp)),*list(range(nimp+asorbs[0],nimp+asorbs[0]+asorbs[1]))]

    # Call pre-SOC impurity solver
    if imp_solver is None:
        pass

    elif imp_solver == 'hf':
        nelec = int(mf.mol.nelectron - asorbs[-1]*2)
        ldm = get_dmet_imp_ldm(mf,imp_inds)
        solver_base = scfsol.rohf_solve_imp(as1e,as2e,nelec,mf.mol.spin,ssdmet.max_mem,dm0=ldm)

        if not ssdmet.silent:
            with open(ssdmet.title+'_imp_rohf_orbs.molden', 'w') as f1:
                molden.header(mf.mol, f1)
                molden.orbital_coeff(mf.mol, f1, caoas[:,act_inds] @ solver_base.mo_coeff, ene=solver_base.mo_energy, occ=solver_base.mo_occ)

    elif imp_solver == 'casscf':
        caschk_fname = ssdmet.title + '_imp_cas_chk.h5'
        if not os.path.isfile(caschk_fname):
            nelec = int(mf.mol.nelectron - asorbs[-1]*2)
            ldm = get_dmet_imp_ldm(mf,imp_inds)
            solver_base = scfsol.rohf_solve_imp(as1e,as2e,nelec,mf.mol.spin,ssdmet.max_mem,dm0=ldm)

            if not ssdmet.silent:
                with open(ssdmet.title+'_imp_rohf_orbs.molden', 'w') as f1:
                    molden.header(mf.mol, f1)
                    molden.orbital_coeff(mf.mol, f1, caoas[:,act_inds] @ solver_base.mo_coeff, ene=solver_base.mo_energy, occ=solver_base.mo_occ)

            casinfo_fname = ssdmet.title + '_cas_info'

            if not os.path.isfile(casinfo_fname):
                logger.error(ssdmet,'Failed to read saved CAS briefings.')
                exit()

            ncasorb,ncaselec,casorbind = read_cas_info(casinfo_fname)
            solver = mcscfsol.sacasscf_solve_imp(solver_base,mf.mol,ncasorb,ncaselec,casorbind,statelis)

            mcscfsol.sacasscf_dump_chk(solver,caschk_fname)
        else:
            solver = mcscfsol.sacasscf_load_chk(lib.StreamObject(),caschk_fname)


    # Call SOC impurity solver
    if imp_solver_soc is None:
        pass

    elif imp_solver_soc == 'siso':
        mysiso = siso.SISO(solver,statelis,ssdmet.title,mf.mol,caoas,asorbs)
        mysiso.kernel() 

        ssdmet.mag_energy = mysiso.mag_energy
        ssdmet.opt_energy = solver.e_states

        Ha2cm = 219474.63

        np.savetxt(ssdmet.title+'_opt.txt',(solver.e_states-np.min(solver.e_states))*Ha2cm,fmt='%.6f')
        np.savetxt(ssdmet.title+'_mag.txt',(mysiso.mag_energy-np.min(mysiso.mag_energy))*Ha2cm,fmt='%.6f')

    return 0

class SSDMET(lib.StreamObject):
    """
    single-shot DMET with cluster-environment partition
    """

    silent = False
    
    def __init__(self,mf,title='untitled',max_mem=64000,verbose=logger.INFO):
        self.mf = mf
        self.title = title
        self.max_mem = max_mem
        self.verbose = verbose

    def kernel(self,imp_inds,imp_solver=None,imp_solver_soc=None,statelis=None,thres=1e-13):
        return kernel(self,imp_inds,imp_solver,imp_solver_soc,statelis,thres)

