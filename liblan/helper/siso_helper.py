from liblan.soc import siso
from liblan.solver import mcscfsol
from liblan.dmet import ssdmet

from pyscf import gto, scf, mcscf, lib
from pyscf.mcscf import avas 
from pyscf.tools import molden
from pyscf.lib import logger

import os
import numpy as np

def cassi_so(mol,statelis,aslabel,title,select='avas',chk=None):
    # AVAS for active space (manual OK)
    # aslabel : ['Co 3d']

    mf = scf.rohf.ROHF(mol).x2c()
    if chk is None:
        chk_fname = title + '_rohf.chk'
    else:
        chk_fname = chk

    mf.chkfile = chk_fname
    mf.init_guess = 'chk'
    mf.level_shift = .2
    mf.max_cycle = 1
    mf.max_memory = 1024*500
    mf.kernel()

    if select == 'manual':
        with open(title+'_rohf_orbs.molden', 'w') as f1:
            molden.header(mol, f1)
            molden.orbital_coeff(mol, f1, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)


    caschk_fname = title + '_cas_chk.h5'
    if not os.path.isfile(caschk_fname):
        if select == 'avas':
            ncasorb,ncaselec,casorbind = avas.avas(mf, aslabel, canonicalize=False)
            mc = mcscfsol.sacasscf_solve_imp(mf,mol,ncasorb,ncaselec,casorbind,statelis,avas=True)
            mcscfsol.sacasscf_dump_chk(mc,caschk_fname)
        elif select == 'manual':
            casinfo_fname = title + '_cas_info'
            if not os.path.isfile(casinfo_fname):
                logger.error(mf,'Failed to read CAS briefings.')
                exit()

            ncasorb,ncaselec,casorbind = ssdmet.read_cas_info(casinfo_fname)
            mc = mcscfsol.sacasscf_solve_imp(mf,mol,ncasorb,ncaselec,casorbind,statelis)
            mcscfsol.sacasscf_dump_chk(mc,caschk_fname)
            with open(title+'_cas_orbs.molden', 'w') as f1:
                molden.header(mol, f1)
                molden.orbital_coeff(mol, f1, mc.mo_coeff, ene=mc.mo_energy, occ=mc.mo_occ)

    else:
        mc = mcscfsol.sacasscf_load_chk(lib.StreamObject(),caschk_fname)
        
    mysiso = siso.SISO(mc,statelis,title,mol)
    mysiso.kernel()

    Ha2cm = 219474.63
    np.savetxt(title+'_opt.txt',(mc.e_states-np.min(mc.e_states))*Ha2cm,fmt='%.6f')
    np.savetxt(title+'_mag.txt',(mysiso.mag_energy-np.min(mysiso.mag_energy))*Ha2cm,fmt='%.6f')

    return 0

