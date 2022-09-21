import numpy as np
import time
import os
import h5py

from sympy.physics.quantum.cg import Wigner3j
from sympy import symbols
from itertools import product
from pyscf import lib, scf
from pyscf.lib import logger 
from pyscf.data import nist
from pyscf.fci import cistring
from functools import reduce

def calc_z(siso,mol,caoas=None,asorbs=None):

    mc = siso.mc

    # 1e SOC integrals
    hso1e = mol.intor('int1e_pnucxp',3)

    if caoas is None or asorbs is None:
        # All electron SISO
        mo_cas = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
        sodm1 = mc.make_rdm1()
    else:
        # DMET SISO
        nimp = caoas.shape[0]-np.sum(asorbs)
        act_inds = [*list(range(nimp)),*list(range(nimp+asorbs[0],nimp+asorbs[0]+asorbs[1]))]
        uos = caoas.shape[0]-asorbs[-1]
        dm_uo = caoas[:,uos:] @ caoas[:,uos:].conj().T*2
        mo_cas = caoas[:,act_inds] @ mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
        sodm1_act = reduce(np.dot,[caoas[:,act_inds],mc.make_rdm1(),caoas[:,act_inds].conj().T])
        sodm1 = sodm1_act+dm_uo

    # 2e SOC J/K1/K2 integrals
    # SOC_2e integrals are anti-symmetric towards exchange (ij|kl) -> (ji|kl)
    vj,vk,vk2 = scf.jk.get_jk(mol,[sodm1,sodm1,sodm1],['ijkl,kl->ij','ijkl,jk->il','ijkl,li->kj'],intor='int2e_p1vxp1',comp=3)
    hso2e = vj - 1.5 * vk - 1.5 * vk2
    
    alpha = nist.ALPHA
    hso = 1.j*(alpha**2/2)*(hso1e+hso2e)
    h1 = np.asarray([reduce(np.dot, (mo_cas.T, x.T, mo_cas)) for x in hso])
    z = np.asarray([1/np.sqrt(2)*(h1[0]-1.j*h1[1]),h1[2],-1/np.sqrt(2)*(h1[0]+1.j*h1[1])])
    return z

def calc_a(siso):
    mc = siso.mc
    Stuples = siso.Stuples

    a = []
    for i, item in enumerate(Stuples):
        if item[0] == item[1]: # SS part
            S = item[0]
            norb = mc.ncas
            nelec = sum(mc.nelecas)
            nelec_a = (nelec+S)//2
            nelec_b = nelec-nelec_a
            a.append([cistring.gen_linkstr_index(list(range(norb)),nelec_a),cistring.gen_linkstr_index(list(range(norb)),nelec_b)])

        elif item[0] + 2 == item[1]: # SS+1 part
            S = item[0] + 2
            norb = mc.ncas
            nelec = sum(mc.nelecas)
            nelec_a = (nelec+S)//2
            nelec_b = nelec-nelec_a
            a.append([cistring.gen_des_str_index(list(range(norb)),nelec_a),cistring.gen_cre_str_index(list(range(norb)),nelec_b)])

        elif item[0] == item[1] + 2: # SS-1 part
            S = item[0] - 2
            norb = mc.ncas
            nelec = sum(mc.nelecas)
            nelec_a = (nelec+S)//2
            nelec_b = nelec-nelec_a
            a.append([cistring.gen_cre_str_index(list(range(norb)),nelec_a),cistring.gen_des_str_index(list(range(norb)),nelec_b)])

        else: # No connection
            a.append(0)
    
    imds = siso.imds
    imds.a = a
    return a

def calc_c(siso):
    statelis = siso.statelis
    Slis = siso.Slis
    mc = siso.mc 

    c = []
    for i in range(len(Slis)):
        try:
            c.append(np.asarray(mc.ci[int(np.sum(statelis[:Slis[i]])):int(np.sum(statelis[:Slis[i]+1]))]))
        except IndexError:
            c.append(np.asarray(mc.ci[int(np.sum(statelis[:Slis[i]])):int(np.sum(statelis))]))
    
    imds = siso.imds
    imds.c = c
    return c
    
def calc_e(siso):
    statelis = siso.statelis
    Slis = siso.Slis
    mc = siso.mc 

    e = []
    for i in range(len(Slis)):
        try:
            e.append(np.asarray(mc.e_states[int(np.sum(statelis[:Slis[i]])):int(np.sum(statelis[:Slis[i]+1]))]))
        except IndexError:
            e.append(np.asarray(mc.e_states[int(np.sum(statelis[:Slis[i]])):int(np.sum(statelis))]))
    
    imds = siso.imds
    imds.e = e
    return e

def calc_d(siso):
    # TODO could try C-based version; with small AS calc_z() takes much more time, however
    Slis = siso.Slis
    Stuples = siso.Stuples
    imds = siso.imds

    d_col = []
    for i, item in enumerate(Stuples):
        if item[0] == item[1]: # SS part

            S = item[0]
            iS = np.where(Slis==S)[0][0]
            z = imds.z 
            c = imds.c 
            a = imds.a[i]

            b = np.zeros((3,*c[iS].shape),dtype='complex')
            for b0 in range(b.shape[0]):
                for b1 in range(b.shape[1]):
                    for b2 in range(b.shape[2]):
                        for b3 in range(b.shape[3]):
                            for aa in range(a[0].shape[1]):
                                b[b0,b1,b2,b3] += c[iS][b1,a[0][b2,aa,2],b3]*z[b0,a[0][b2,aa,0],a[0][b2,aa,1]]*a[0][b2,aa,3]
                            for ab in range(a[1].shape[1]):
                                b[b0,b1,b2,b3] -= c[iS][b1,b2,a[1][b3,ab,2]]*z[b0,a[1][b3,ab,0],a[1][b3,ab,1]]*a[1][b3,ab,3]
            
            g = np.zeros((3,S+1,S+1),dtype='complex')
            ss = symbols('S')
            for g0 in range(g.shape[0]):
                for g1 in range(g.shape[1]):
                    for g2 in range(g.shape[2]):
                        g[g0,g1,g2] += (-1)**(g0+S/2-(g1-S/2))*Wigner3j(ss/2,-(g1-ss/2),1,1-g0,ss/2,g2-ss/2).subs(ss,S).doit()
            w = np.einsum('kij,mnij->mnk',c[iS],b)
            d = np.einsum('mij,mkl->kilj',g,w)

        elif item[0] + 2 == item[1]: # SS+1 part

            S = item[0]
            iS = np.where(Slis==S)[0][0]
            iSp = np.where(Slis==S+2)[0][0]
            z = imds.z 
            c = imds.c 
            a = imds.a[i]

            b = np.zeros((3,c[iS].shape[0],c[iSp].shape[1],c[iSp].shape[2]),dtype='complex')
            for b0 in range(b.shape[0]):
                for b1 in range(b.shape[1]):
                    for b2 in range(b.shape[2]):
                        for b3 in range(b.shape[3]):
                            for aa in range(a[0].shape[1]):
                                for ab in range(a[1].shape[1]):
                                    b[b0,b1,b2,b3] += c[iS][b1,a[0][b2,aa,2],a[1][b3,ab,2]]*z[b0,a[1][b3,ab,0],a[0][b2,aa,1]]*a[0][b2,aa,3]*a[1][b3,ab,3]
            
            g = np.zeros((3,S+1,S+3),dtype='complex')
            ss = symbols('S')
            for g0 in range(g.shape[0]):
                for g1 in range(g.shape[1]):
                    for g2 in range(g.shape[2]):
                        g[g0,g1,g2] += (-1)**(g0+S/2-(g1-S/2))*Wigner3j(ss/2,-(g1-ss/2),1,1-g0,ss/2+1,g2-ss/2-1).subs(ss,S).doit()
                        
            w = np.einsum('kij,mnij->mnk',c[iSp],b)
            d = np.einsum('mij,mkl->kilj',g,w)

        elif item[0] == item[1] + 2: # SS-1 part

            S = item[0]
            iS = np.where(Slis==S)[0][0]
            iSm = np.where(Slis==S-2)[0][0]
            z = imds.z 
            c = imds.c 
            a = imds.a[i]

            b = np.zeros((3,c[iS].shape[0],c[iSm].shape[1],c[iSm].shape[2]),dtype='complex')
            for b0 in range(b.shape[0]):
                for b1 in range(b.shape[1]):
                    for b2 in range(b.shape[2]):
                        for b3 in range(b.shape[3]):
                            for aa in range(a[0].shape[1]):
                                for ab in range(a[1].shape[1]):
                                    b[b0,b1,b2,b3] += c[iS][b1,a[0][b2,aa,2],a[1][b3,ab,2]]*z[b0,a[0][b2,aa,0],a[1][b3,ab,1]]*a[0][b2,aa,3]*a[1][b3,ab,3]
            
            g = np.zeros((3,S+1,S-1),dtype='complex')
            ss = symbols('S')
            for g0 in range(g.shape[0]):
                for g1 in range(g.shape[1]):
                    for g2 in range(g.shape[2]):
                        g[g0,g1,g2] += (-1)**(g0+S/2-(g1-S/2))*Wigner3j(ss/2,-(g1-ss/2),1,1-g0,ss/2-1,g2-ss/2+1).subs(ss,S).doit()
            w = np.einsum('kij,mnij->mnk',c[iSm],b)
            d = np.einsum('mij,mkl->kilj',g,w)


        else: # No connection
            d = 0
        
        d_col.append(d)
    return d_col

def calc_y(siso):
    Slis = siso.Slis
    Stuples = siso.Stuples
    imds = siso.imds

    y_col = []
    for i, item in enumerate(Stuples):
        if item[0] == item[1]: # SS part

            S = item[0]
            iS = np.where(Slis==S)[0][0]
            z = imds.z 
            c = imds.c 
            a = imds.a[i]

            b = np.zeros((3,*c[iS].shape),dtype='complex')
            for b0 in range(b.shape[0]):
                for b1 in range(b.shape[1]):
                    for b2 in range(b.shape[2]):
                        for b3 in range(b.shape[3]):
                            for aa in range(a[0].shape[1]):
                                b[b0,b1,b2,b3] += c[iS][b1,a[0][b2,aa,2],b3]*z[b0,a[0][b2,aa,0],a[0][b2,aa,1]]*a[0][b2,aa,3]
                            for ab in range(a[1].shape[1]):
                                b[b0,b1,b2,b3] -= c[iS][b1,b2,a[1][b3,ab,2]]*z[b0,a[1][b3,ab,0],a[1][b3,ab,1]]*a[1][b3,ab,3]
            

            w = np.einsum('kij,mnij->mnk',c[iS],b)
            y = w / 2

        elif item[0] + 2 == item[1]: # SS+1 part

            S = item[0]
            iS = np.where(Slis==S)[0][0]
            iSp = np.where(Slis==S+2)[0][0]
            z = imds.z 
            c = imds.c 
            a = imds.a[i]

            b = np.zeros((3,c[iS].shape[0],c[iSp].shape[1],c[iSp].shape[2]),dtype='complex')
            for b0 in range(b.shape[0]):
                for b1 in range(b.shape[1]):
                    for b2 in range(b.shape[2]):
                        for b3 in range(b.shape[3]):
                            for aa in range(a[0].shape[1]):
                                for ab in range(a[1].shape[1]):
                                    b[b0,b1,b2,b3] += c[iS][b1,a[0][b2,aa,2],a[1][b3,ab,2]]*z[b0,a[1][b3,ab,0],a[0][b2,aa,1]]*a[0][b2,aa,3]*a[1][b3,ab,3]
                        
            w = np.einsum('kij,mnij->mnk',c[iSp],b)
            y = w / np.sqrt(2)


        elif item[0] == item[1] + 2: # SS-1 part

            S = item[0]
            iS = np.where(Slis==S)[0][0]
            iSm = np.where(Slis==S-2)[0][0]
            z = imds.z 
            c = imds.c 
            a = imds.a[i]

            b = np.zeros((3,c[iS].shape[0],c[iSm].shape[1],c[iSm].shape[2]),dtype='complex')
            for b0 in range(b.shape[0]):
                for b1 in range(b.shape[1]):
                    for b2 in range(b.shape[2]):
                        for b3 in range(b.shape[3]):
                            for aa in range(a[0].shape[1]):
                                for ab in range(a[1].shape[1]):
                                    b[b0,b1,b2,b3] += c[iS][b1,a[0][b2,aa,2],a[1][b3,ab,2]]*z[b0,a[0][b2,aa,0],a[1][b3,ab,1]]*a[0][b2,aa,3]*a[1][b3,ab,3]
            
            w = np.einsum('kij,mnij->mnk',c[iSm],b)
            y = -w / np.sqrt(2) 


        else: # No connection
            y = 0
        
        # spherical -> cartesian

        yc = np.asarray([(y[0]-y[2])/np.sqrt(2),1.j*(y[2]+y[0])/np.sqrt(2),y[1]])
        y_col.append(yc)
    return y_col

def calc_h(siso):
    statelis = siso.statelis
    Slis = siso.Slis
    Stuples = siso.Stuples
    imds = siso.imds 

    d = imds.d
    e = imds.e

    h_col = [[0]*len(Slis) for i in range(len(Slis))]

    for i in range(len(Slis)):
        for j in range(len(Slis)):
            s1, s2 = Slis[i], Slis[j]
            indt = Stuples.index((s1,s2))

            if s1 == s2: # SS part
                S = s1
                counter = [(i+1)*(x) for i,x in enumerate(statelis)]
                nstates = counter[S]
                coeff = np.sqrt((S/2+1)*(S+1)/(S/2))/2
                h = coeff * d[indt].reshape((nstates,nstates),order='C')

                for ns in range(nstates):
                    n1,m1 = divmod(ns,S+1)
                    h[ns,ns] += e[i][n1]

            elif s1 + 2 == s2: # SS+1 part
                S = s1
                counter = [(i+1)*(x) for i,x in enumerate(statelis)]
                nstatesa = counter[S]
                nstatesb = counter[S+2]
                coeff = np.sqrt((S+3)/2)
                h = coeff * d[indt].reshape((nstatesa,nstatesb),order='C')

            elif s1 == s2 + 2: # SS-1 part
                S = s1
                counter = [(i+1)*(x) for i,x in enumerate(statelis)]
                nstatesa = counter[S]
                nstatesb = counter[S-2]
                coeff = -np.sqrt((S+1)/2)
                h = coeff * d[indt].reshape((nstatesa,nstatesb),order='C')

            else:
                counter = [(i+1)*(x) for i,x in enumerate(statelis)]
                nstatesa = counter[s1]
                nstatesb = counter[s2]
                h = np.zeros((nstatesa,nstatesb),dtype='complex')

            h_col[i][j] = h
    
    return np.block(h_col)

def build_imds(siso,mol,caoas=None,asorbs=None):
    imds = siso.imds
    cput0 = (time.process_time(), time.perf_counter())
    z_fname = siso.title + '_siso_z.h5'
    if not os.path.isfile(z_fname):
        imds.z = calc_z(siso,mol,caoas,asorbs)
        with h5py.File(z_fname, 'w') as fh5:
            fh5['z'] = imds.z
    else:
        with h5py.File(z_fname, 'r') as fh5:
            imds.z = fh5['z'][:]
    
    cput1 = logger.timer(siso, 'SISO z amplitudes', *cput0)

    d_fname = siso.title + '_siso_d.h5'
    if not os.path.isfile(d_fname):
        imds.a = calc_a(siso)
        imds.c = calc_c(siso)
        imds.e = calc_e(siso)
        imds.d = calc_d(siso)

        with h5py.File(d_fname, 'w') as fh5:
            for s in range(len(imds.d)):
                fh5['d'+str(s)] = imds.d[s]
    else:
        with h5py.File(d_fname, 'r') as fh5:
            imds.d = [fh5['d'+str(s)][:] for s in range(len(siso.Stuples))]
            imds.e = calc_e(siso)
            imds.c = calc_c(siso)
            imds.a = calc_a(siso)

    cput2 = logger.timer(siso, 'SISO d amplitudes', *cput1)

    imds.y = calc_y(siso)    

def kernel(siso,mol=None,caoas=None,asorbs=None,method='qdpt'):
    """
    Driver function for SI-SO
    """

    siso.build_imds(mol,caoas,asorbs)

    if method == 'qdpt':
        h = siso.calc_h()
        np.savetxt('h.txt',h,fmt='%.6f')

        flag = np.allclose(h,np.conj(h.T))
        logger.info(siso,'Whether h is Hermitian: %s',flag)

        siso.mag_energy,siso.mag_coeffs = np.linalg.eigh(h)
        logger.info(siso,'Selected mag energies (cm^-1): \n %s',(siso.mag_energy[:10]-min(siso.mag_energy))*219474.63)
    elif method == 'pt2':
        # TODO clean up
        
        S = np.max(siso.Slis)
        Stuples = siso.Stuples
        d = np.zeros((3,3),dtype='complex')
        e = siso.imds.e
        Y = siso.imds.y
        ge = e[-1][0]

        print(e[-1])

        for i in range(3):
            for j in range(3):
                # SS part
                ind = Stuples.index((S,S))
                d[i,j] += -np.einsum('i,i,i->',1/(e[-1][1:]-ge),Y[ind][i,0,1:],Y[ind][j,1:,0])/(S/2)**2

                # SS-1 part
                ind1 = Stuples.index((S,S-2))
                ind2 = Stuples.index((S-2,S))
                d[i,j] += -np.einsum('i,i,i->',1/(e[-2]-ge),Y[ind1][i,0,:],Y[ind2][j,:,0])/((S/2)*(S-1))

        print(d)
        ev, evc = np.linalg.eigh(d)

        print(ev*219474.63)
        exit()
        
    return siso

class _IMDS():
    """
    SI-SO intermediates

    self.z      :   (3,nao,nao)
    self.a      :   (nSt,2,ncia||b,4)
    self.b
    self.c      :   (nS,nstates,ncia,ncib)
    self.e      :   (nS,nstates)
    """
    def __init__(self):
        self.z = None 

class SISO(lib.StreamObject):
    def __init__(self,mc,statelis,title,mol=None,caoas=None,asorbs=None,verbose=logger.INFO):
        self.mc = mc
        self.statelis = statelis
        self.title = title

        self.mol = mol 
        self.caoas = caoas 
        self.asorbs = asorbs

        self.imds = _IMDS()
        self.Slis = np.nonzero(self.statelis)[0]
        self.Stuples = [x for x in product(self.Slis,self.Slis)]

        self.verbose = verbose

    def build_imds(self,mol=None,caoas=None,asorbs=None):
        return build_imds(self,mol,caoas,asorbs)

    def calc_h(self):
        return calc_h(self)

    def kernel(self,mol=None,caoas=None,asorbs=None):
        if mol is None:
            mol = self.mol 
        if caoas is None:
            caoas = self.caoas 
        if asorbs is None:
            asorbs = self.asorbs

        return kernel(self,mol,caoas,asorbs)




