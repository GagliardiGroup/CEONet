import glob
import re
from pyscf import gto, scf
from pyscf.tools import molden
import numpy as np

def extract_modct(mol,mos):
    """
    Returns
    -------
    all_dct : dict
        Keys are angular momentum l (0 for s, 1 for p, etc.)
        Values are arrays of shape (n_prim, 2 + dim + 1), where each row is:
        [exponent, coeff, ao_idx_1, ..., ao_idx_N, atom_idx]
    """
    #Pulls out [exp coeff 13 14 15 16 17 18 N] for each orbital
    #Note that prim coeffs are radial normalized (see https://pyscf.org/pyscf_api_docs/pyscf.gto.html) and contraction-normalized
    #Atomic orbitals are not angular normalized, but these are pure functions of l so the model can adapt
    size_dct = {0:1,1:3,2:6,3:10,4:15}
    all_dct = {}
    ao_start = 0
    for shell in mol._bas:
        #ncontract != 1 for bases with shared exps (e.g. cc-pvdz)
        atm_idx, l, nprim, ncontract, _, exp_start, coeff_start, _ = shell
        exps = mol._env[exp_start:exp_start+nprim]
        for n in range(ncontract):
            coeffs = mol._env[(coeff_start+nprim*n):(coeff_start+nprim*n+nprim)]
            ints = np.hstack([np.arange(ao_start,ao_start + size_dct[l]),np.array([atm_idx])])
            ints = np.vstack([ints]*len(exps))
            arr = np.hstack([exps[:,None],coeffs[:,None],ints])
            if l not in all_dct.keys():
                all_dct[l] = []
            all_dct[l].append(arr)
            ao_start += size_dct[l]
    
    for l in all_dct.keys():
        all_dct[l] = np.vstack(all_dct[l])
    
    #Sanity checks
    #Asserts that we have expected number of primitives and all aos represented
    #Can take out for very minor speedup
    nprim = mol._bas[:,2] * mol._bas[:,3]
    n_lst = []
    for l in np.unique(mol._bas[:,1]):
        idx = np.where(mol._bas[:,1] == l)
        n_lst += [np.unique(all_dct[l][:,2:-1])]
        assert(nprim[idx].sum() == all_dct[l].shape[0])
    n_lst = np.sort(np.hstack(n_lst))
    assert(np.allclose(n_lst,np.arange(mol.nao)))
    
    return all_dct

class OrbExtract():
    def __init__(self, fn=None, rotate=False, cart=True, basis_fill="sto3g",
                 mol=None, mo_ene=None, mo_coeff=None, mo_occ=None, labels=None,
                ):
        if mol is None: #else fn is none
            mol, mo_ene, mo_coeff, mo_occ, _, _ = molden.load(fn)

        #Project onto cartesian for l > 1
        if cart:
            mol2 = mol.copy()
            mol2.cart = True
            mo_coeff = scf.addons.project_mo_nr2nr(mol,mo_coeff,mol2)
            mol = mol2

        #Assign values IN BOHR
        self.mol = mol
        self.els = mol._atm[:,0]
        self.xyz = np.vstack([atm[1] for atm in mol._atom])
        self.mo_ene = mo_ene
        self.mo_occ = mo_occ
        self.labels = labels
        
        self.nmos = mo_coeff.shape[1]
        self.rotate = rotate
        self.dct = extract_modct(mol,mo_coeff)
        self.mo_coeff = mo_coeff
        if mo_occ is not None:
            self.charge = int(mol.nelectron - sum(mo_occ))
        else:
            self.charge = None
        if mol.basis == {}:
            mol.basis = basis_fill
        
        if rotate:
            # self.orb_dct = assign_mos(self.dct,mo_coeff)
            print("Recomputing rotated...")
            from scipy.spatial.transform import Rotation as R
            rmol = gto.Mole()
            els = mol.elements
            coords = np.vstack([atm[1] for atm in mol.atom])
            r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]) #90 around z
            coords = (r.as_matrix() @ coords.T).T
            rmol.atom = [[el,coord] for el,coord in zip(els,coords)]
            rmol.charge = self.charge
            rmol.basis = mol.basis
            rmol.build()
            # print(mol._basis.keys(), rmol._basis.keys())
            rmf = scf.RHF(rmol)
            rmf.kernel()
            #Need to also project this for comparison lol
            if cart:
                rmol2 = rmol.copy()
                rmol2.cart = True
                rmo_coeff = scf.addons.project_mo_nr2nr(rmol,rmf.mo_coeff,rmol2)
            else:
                rmo_coeff = rmf.mo_coeff
            self.rmo_coeff = rmo_coeff
            self.rxyz = np.vstack([atm[1] for atm in rmol._atom])
            self.rdct = extract_modct(mol,rmo_coeff)
            # self.rorb_dct = assign_mos(self.rdct,rmo_coeff)
    
            #Check for consistency
            # check1 = diff(self.orb_dct[0],self.rorb_dct[0])
            # check2 = diff(self.orb_dct[1][:,:,0],self.rorb_dct[1][:,:,1])
            # check3 = diff(self.orb_dct[1][:,:,1],self.rorb_dct[1][:,:,0])
            # check4 = diff(self.orb_dct[1][:,:,2],self.rorb_dct[1][:,:,2])
            # check5 = diff(mo_ene,rmf.mo_energy)
            # check6 = diff(mo_occ,rmf.mo_occ)
            # for i,c in enumerate([check1,check2,check3,check4,check5,check6]):
            #     assert(c < 1e-7)

    def extract_nlm(self,mo_num,rotate=False):
        orbdct = self.dct
        xyz = self.xyz
        mos = self.mo_coeff
        if rotate:
            orbdct = self.rdct
            xyz = self.rxyz
            mos = self.rmo_coeff
            orbdct = self.rdct
        dct = {
            "atomic_numbers":self.els.astype("uint8"),
            "positions":xyz.astype("float32"),
        }
        if self.mo_ene is not None:
            dct["energy"] = self.mo_ene[[mo_num]].astype("float32")
        if self.mo_occ is not None:
            dct["occ"] = self.mo_occ[[mo_num]].astype("uint8"),
        if self.labels is not None:
            dct["labels"] = self.labels[[mo_num]]
        if self.charge is not None:
            dct["charge"] = np.array([self.charge]).astype("int8"),
        
        c = mos[:,mo_num]
        dct["c"] = c.astype("float32")
        
        for l,v in orbdct.items(): #max 65k orbs from uint16
            dct[f"orbints_{l}"] = orbdct[l][:,2:].astype("uint16")
            dct[f"orbfloats_{l}"] = orbdct[l][:,:2].astype("float32")
        return dct

def diff(arr1,arr2):
    return np.abs(np.abs(arr1) - np.abs(arr2)).max()

def assign_mos(dct,mos):
    #doesn't really work for l > 1
    #But works as a sanity check on the l=0 and l=1 features
    mo_dct = {}
    for l in dct.keys():
        orb_ints = dct[l][:,2:-1]
        np = orb_ints.shape[0]
        l_len = orb_ints.shape[1]
        orb_c = mos[orb_ints.ravel().astype(int),:]
        orb_c = orb_c.T.reshape(-1,np,l_len)
        mo_dct[l] = orb_c
    return mo_dct