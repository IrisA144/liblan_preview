# Regularized DIIS technique
# Reg terms could be bare polarization, dS, ...

import numpy as np 
from pyscf.lib import logger
from pyscf.scf import diis
from pyscf import scf, lib, gto
from pyscf.scf.rohf import get_roothaan_fock
from pyscf.scf import hf
import scipy.linalg as linalg

from liblan.dmet import ssdmet
from functools import reduce



# Use this to change HF/KS objects into RDIIS-assisted ones
def tag_rdiis_(mf,reg='dS',imp_inds=None,power=.2):

    def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):

        if h1e is None: h1e = mf.get_hcore()
        if s1e is None: s1e = mf.get_ovlp()
        if vhf is None: vhf = mf.get_veff(mf.mol, dm)
        if dm is None: dm = mf.make_rdm1()
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            dm = np.array((dm*.5, dm*.5))

        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        f = get_roothaan_fock((focka,fockb), dm, s1e)
        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f

        if diis_start_cycle is None:
            diis_start_cycle = mf.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = mf.level_shift
        if damp_factor is None:
            damp_factor = mf.damp

        dm_tot = dm[0] + dm[1]
        if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4:
            raise NotImplementedError('ROHF Fock-damping')
        if diis and cycle >= diis_start_cycle:
            f = diis.update(s1e, dm_tot, f, [dm[0],dm[1]], mf, h1e, vhf)
        if abs(level_shift_factor) > 1e-4:
            f = hf.level_shift(s1e, dm_tot*.5, f, level_shift_factor)
        f = lib.tag_array(f, focka=focka, fockb=fockb)
        return f

    class RDIIS(lib.diis.DIIS):
        def __init__(self, mf=None, filename=None):
            lib.diis.DIIS.__init__(self, mf, filename)
            self.rollback = False
            self.space = 8
            self.reg = reg 
            self.imp_inds = imp_inds

        def get_err_vec1(self, s, d, f, dm):
            '''error vector = SDF - FDS + R'''
            if isinstance(f, np.ndarray) and f.ndim == 2:
                sdf = reduce(np.dot, (s,d,f))
                errvec = sdf.T.conj() - sdf

                vmf = lib.StreamObject()
                vmf.make_rdm1 = lambda *args: (dm[0],dm[1])
                vmf.get_ovlp = lambda *args: s 
                vmf.mol = gto.M()
                vmf.mol.spin = 1

                if self.reg == 'dS':
                    a,ent,b = ssdmet.get_dmet_as_props(vmf,self.imp_inds)
                    logger.info(self, '----------SDDIS-Entropy %.3f', ent)
                    if np.abs(ent) > 0.1:
                        errvec = linalg.block_diag(errvec,power*np.abs(ent))
                    else:
                        errvec = linalg.block_diag(errvec,0)
                elif self.reg == 'P':
                    pass

            elif isinstance(f, np.ndarray) and f.ndim == 3 and s.ndim == 3:
                errvec = []
                for i in range(f.shape[0]):
                    sdf = reduce(np.dot, (s[i], d[i], f[i]))
                    errvec.append((sdf.T.conj() - sdf))
                errvec = np.vstack(errvec)

            elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
                nao = s.shape[-1]
                s = lib.asarray((s,s)).reshape(-1,nao,nao)
                return get_err_vec1(s, d.reshape(s.shape), f.reshape(s.shape),dm)
            else:
                raise RuntimeError('Unknown SCF DIIS type')
            return errvec

        def update(self, s, d, f, dm, *args, **kwargs):
            errvec = self.get_err_vec1(s, d, f, dm)
            logger.debug1(self, 'diis-norm(errvec)=%g', np.linalg.norm(errvec))
            xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
            if self.rollback > 0 and len(self._bookkeep) == self.space:
                self._bookkeep = self._bookkeep[-self.rollback:]
            return xnew

    mf.get_fock = lambda *args: get_fock(mf,*args) 
    mf.DIIS = RDIIS
    return mf


