import glob
import re
from pyscf import gto, scf
from pyscf.tools import molden
import numpy as np

def extract_modct(mol,mos):
    #Cartesian sizes:
    aotyp_dct = {0:"s",1:"p",2:"d",3:"f",4:"g"}
    size_dct = {0:1,1:3,2:6,3:10,4:15}
    aotyps = np.array([s.split()[-1] for s in mol.ao_labels()])
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
        #Some bases have shared weights w/ different alpha, not implemented
        #Should throw an error when stacking above though
        assert(orb_floats.shape[1] == 2)

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

        #Example for d orbital:
        #alpha w 13 14 15 16 17 18 N
    
    return all_dct

class OrbExtract():
    def __init__(self, fn=None, rotate=False, cart=True, label_hl=True,
                 mol=None, mo_ene=None, mo_coeff=None, mo_occ=None,
                ):
        if mol is None: #else fn is none
            mol, mo_ene, mo_coeff, mo_occ, _, _ = molden.load(fn)

        #Project onto cartesian for l > 1
        if cart:
            mol2 = mol.copy()
            mol2.cart = True
            mo_coeff = scf.addons.project_mo_nr2nr(mol,mo_coeff,mol2)
            mol = mol2

        #Assign values in Bohr
        self.mol = mol
        self.els = mol._atm[:,0]
        self.xyz = np.vstack([atm[1] for atm in mol._atom])
        self.mo_ene = mo_ene
        self.mo_occ = mo_occ

        self.label_hl = label_hl
        if self.label_hl:
            occ_mos = np.where(mo_occ == 2)[0]
            virt_mos = np.where(mo_occ == 0)[0]
            homo_ene = np.max(mo_ene[occ_mos])
            lumo_ene = np.min(mo_ene[virt_mos])
            self.is_homo = (np.abs(mo_ene - homo_ene) < 1e-4) * (mo_occ == 2)
            self.is_lumo = (np.abs(mo_ene - lumo_ene) < 1e-4) * (mo_occ == 0)
        
        self.nmos = mo_coeff.shape[1]
        self.rotate = rotate
        self.dct = extract_modct(mol,mo_coeff)
        self.mo_coeff = mo_coeff
        self.charge = int(mol.nelectron - sum(mo_occ))
        
        if rotate:
            self.orb_dct = assign_mos(self.dct,mo_coeff)
            print("Recomputing rotated...")
            from scipy.spatial.transform import Rotation as R
            rmol = gto.Mole()
            els = [atm[0] for atm in mol.atom]
            coords = np.vstack([atm[1] for atm in mol.atom])
            r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]) #90 around z
            coords = (r.as_matrix() @ coords.T).T
            rmol.atom = [[el,coord] for el,coord in zip(els,coords)]
            rmol.charge = self.charge
            rmol.build()
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
            "energy":self.mo_ene[[mo_num]].astype("float32"),
            "occ":self.mo_occ[[mo_num]].astype("uint8"),
            "charge":np.array([self.charge]).astype("int8"),
        }
        if self.label_hl:
            dct["is_homo"] = np.array(self.is_homo[mo_num])
            dct["is_lumo"] = np.array(self.is_lumo[mo_num])
        
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