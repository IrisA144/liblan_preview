from liblan.utils import rdiis
from pyscf import gto, scf

def get_mol(dihedral):
     mol = gto.M(atom = '''
                Co             
                S                  1            2.30186590
                S                  1            2.30186590    2            109.47122060
                S                  1            2.30186590    3            109.47122065    2            -120.00000001                  0
                S                  1            2.30186590    4            109.47122060    3            120.00000001                   0
                H                  2            1.30714645    1            109.47121982    4            '''+str(-60-dihedral)+'''      0
                H                  4            1.30714645    1            109.47121982    3            '''+str(60+dihedral)+'''       0
                H                  5            1.30714645    1            109.47121982    4            '''+str(-180+dihedral)+'''     0
                H                  3            1.30714645    1            109.47121982    4            '''+str(60-dihedral)+'''       0
     ''',
     basis = 'def2tzvp', symmetry=0 ,spin = 3,charge = -2,verbose= 4)

     return mol

mol = get_mol(0)

mf = scf.rohf.ROHF(mol).x2c()
rdiis.tag_rdiis_(mf,imp_inds=mol.search_ao_label('Co *'),power=.2)
mf.level_shift = .2
mf.chkfile = 'rdiis.chk'
mf.diis_space = 12
mf.max_memory = 8000
mf.max_cycle = 1000

mf.kernel()

"""
DIIS:           E: -2986.17     S: 0.720        24 cycles
R-DIIS-0.1:     E: -2986.43     S: 0.024        202 cycles
R-DIIS-0.2:     E: -2986.43     S: 0.024        84 cycles
R-DIIS-0.3:     E: -2986.43     S: 0.024        74 cycles
"""