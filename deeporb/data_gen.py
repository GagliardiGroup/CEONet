import glob
import re
from pyscf import gto, scf
from pyscf.tools import molden
import numpy as np

def diff(arr1,arr2):
    return np.abs(np.abs(arr1) - np.abs(arr2)).max()

def extract_modct(mol,mos):
    aotyp_dct = {0:"s",1:"p",2:"d"} #tbd: implement d orbs and double zeta
    size_dct = {0:1,1:3,2:6}
    aotyps = np.array([s.split()[-1] for s in mol.ao_labels()])
    bas_info = list(mol._basis.values())
    bas_info = []
    for v in mol._basis.values():
        bas_info += v
    all_dct = {}
    
    for l in aotyp_dct.keys():
        l_idx = [i for i,s in enumerate(aotyps) if aotyp_dct[l] in s]
        if len(l_idx) == 0:
            continue
        
        #alpha w
        bas = [v for v in bas_info if v[0] == l] #lao x np x 2
        bas = [np.vstack(v[1:]) for v in bas] #lao x np x 2
        orb_floats = np.vstack(bas) #(lao x np) x 2

        #i a
        nums = mol._bas[:,0][np.where(mol._bas[:,1] == l)[0]] #lao
        c = np.array(l_idx).reshape(-1,size_dct[l]) #lao x dim
        orb_ints = []
        for i,b in enumerate(bas): #lao times
            c2 = np.stack([c[i]] * b.shape[0],axis=0) #np x dim
            n2 = np.stack([nums[i]] * b.shape[0],axis=0)[:,None] #np x 1            
            orb_ints.append(np.hstack([c2,n2]))
        orb_ints = np.vstack(orb_ints) #--> (lao x np) x (dim+1)
        
        out = np.concatenate([orb_floats,orb_ints],axis=-1) #(lao x np) x (2+dim+1)
        all_dct[l] = out
    
    return all_dct

def assign_mos(dct,mos):
    mo_dct = {}
    for l in dct.keys():
        orb_ints = dct[l][:,2:-1]
        np = orb_ints.shape[0]
        l_len = orb_ints.shape[1]
        orb_c = mos[orb_ints.ravel().astype(int),:]
        orb_c = orb_c.T.reshape(-1,np,l_len)
        mo_dct[l] = orb_c
    return mo_dct

class OrbExtract():
    def __init__(self, fn, rotate=False, invert=False):
        mol, mo_ene, mo_coeff, mo_occ, _, _ = molden.load(fn)
        self.mol = mol
        self.name = fn.split("/")[-1].split(".")[0]
        self.nmos = mo_coeff.shape[1]
        self.rotate = rotate
        self.dct = extract_modct(mol,mo_coeff)
        self.orb_dct = assign_mos(self.dct,mo_coeff)
        self.mo_coeff = mo_coeff

        if rotate:
            print("Recomputing rotated...")
            from scipy.spatial.transform import Rotation as R
            rmol = gto.Mole()
            els = [atm[0] for atm in mol.atom]
            coords = np.vstack([atm[1] for atm in mol.atom])
            r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]) #90 around z
            coords = (r.as_matrix() @ coords.T).T
            rmol.atom = [[el,coord] for el,coord in zip(els,coords)]
            rmol.build()
            rmf = scf.RHF(rmol)
            rmf.kernel()
            if self.compute_fock:
                self.rfock = rmf.get_fock()
            rmo_coeff = rmf.mo_coeff
            self.rmo_coeff = rmo_coeff
            self.rxyz = np.vstack([atm[1] for atm in rmol._atom]) #LOL xD
            self.rdct = extract_modct(mol,rmo_coeff)
            self.rorb_dct = assign_mos(self.rdct,rmo_coeff)
    
            #Check for consistency
            check1 = diff(self.orb_dct[0],self.rorb_dct[0])
            check2 = diff(self.orb_dct[1][:,:,0],self.rorb_dct[1][:,:,1])
            check3 = diff(self.orb_dct[1][:,:,1],self.rorb_dct[1][:,:,0])
            check4 = diff(self.orb_dct[1][:,:,2],self.rorb_dct[1][:,:,2])
            check5 = diff(mo_ene,rmf.mo_energy)
            check6 = diff(mo_occ,rmf.mo_occ)
            for i,c in enumerate([check1,check2,check3,check4,check5,check6]):
                assert(c < 1e-7)

        if invert:
            print("Recomputing inverted...")
            imol = gto.Mole()
            els = [atm[0] for atm in mol.atom]
            coords = -np.vstack([atm[1] for atm in mol.atom])
            imol.atom = [[el,coord] for el,coord in zip(els,coords)]
            imol.build()
            imf = scf.RHF(imol)
            imf.kernel()
            imo_coeff = imf.mo_coeff
            self.imo_coeff = imo_coeff
            self.ixyz = np.vstack([atm[1] for atm in imol._atom]) #LOL xD
            self.idct = extract_modct(mol,imo_coeff)
            self.iorb_dct = assign_mos(self.idct,imo_coeff)
    
            #Check for consistency
            check1 = diff(self.orb_dct[0],self.iorb_dct[0])
            check2 = diff(self.orb_dct[1][:,:,0],self.iorb_dct[1][:,:,0])
            check3 = diff(self.orb_dct[1][:,:,1],self.iorb_dct[1][:,:,1])
            check4 = diff(self.orb_dct[1][:,:,2],self.iorb_dct[1][:,:,2])
            check5 = diff(mo_ene,imf.mo_energy)
            check6 = diff(mo_occ,imf.mo_occ)
            for i,c in enumerate([check1,check2,check3,check4,check5,check6]):
                assert(c < 1e-7)

        #Extra info
        self.els = mol._atm[:,0]
        self.xyz = np.vstack([atm[1] for atm in mol._atom])
        self.mo_ene = mo_ene
        self.mo_occ = mo_occ

    def extract_nlm(self,mo_num,rotate=False,invert=False):
        orbdct = self.dct
        xyz = self.xyz
        mos = self.mo_coeff
        if rotate:
            orbdct = self.rdct
            xyz = self.rxyz
            mos = self.rmo_coeff
            orbdct = self.rdct
        if invert:
            orbdct = self.idct
            xyz = self.ixyz
            mos = self.imo_coeff
            orbdct = self.idct
        dct = {
            "atomic_numbers":self.els.astype("uint8"),
            "positions":xyz.astype("float32"),
            "energy":self.mo_ene[[mo_num]].astype("float32"),
            "occ":self.mo_occ[[mo_num]].astype("uint8"),
        }
        
        c = mos[:,mo_num]
        dct["c"] = c.astype("float32")
        
        for l,v in orbdct.items(): #max 65k orbs from uint16
            dct[f"orbints_{l}"] = orbdct[l][:,2:].astype("uint16")
            dct[f"orbfloats_{l}"] = orbdct[l][:,:2].astype("float32")
        return dct