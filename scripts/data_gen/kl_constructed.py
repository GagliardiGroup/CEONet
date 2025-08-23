from dsk.pickle import load_pkl
import numpy as np
from tqdm import tqdm
from deeporb.data_gen import OrbExtract
import os
import glob
import h5py

class Tm_Obj:
    def __init__(self,molden_fn):
        mol, mo_ene, mo_coeff, mo_occ, _, _ = molden.load(molden_fn)
        name = molden_fn.split("/")[-1].split(".")[0]
        labelfn = f"../../../data/kl_constructed/labels/{name}.pkl"
        labels = load_pkl(labelfn)
        self.labelfn = labelfn
        self.molden_fn = molden_fn
        self.labels = labels
        assert(mo_coeff.shape[1] == len(labels))

        #Get rid of core orbitals
        idx = np.array([i for i,l in enumerate(labels) if l != 'Metal core orbital'])
        mo_coeff = mo_coeff[:,idx]
        labels = np.array([l for i,l in enumerate(labels) if l != 'Metal core orbital'])
        
        idx = np.array([i for i,l in enumerate(labels) if l != 'Ligand core orbital'])
        mo_coeff = mo_coeff[:,idx]
        labels = np.array([l for i,l in enumerate(labels) if l != 'Ligand core orbital'])
        self.labels = labels

        dct_to_l = {
            'Ligand valence orbital': 0,
            'Metal valence orbital': 1,
            'Metal-ligand bonding orbital':2,
            'Metal-ligand antibonding orbital':3,
        }
        num_labels = [dct_to_l[l] for l in labels]
        idx_dct = {i:[k for k,j in enumerate(num_labels) if j == i] for i in dct_to_l.values()}

        #Sample evenly from everything
        np.random.seed(34)
        num_to_sample = len(idx_dct[2])
        for i in [0,1]:
            np.random.shuffle(idx_dct[i])
            idx_dct[i] = idx_dct[i][:num_to_sample]
        self.idx_dct = idx_dct

        #Write everything to disk as h5...
        idx = np.hstack(list(idx_dct.values()))
        if len(idx) > 0:
            labels = np.hstack([(np.ones(len(idx_dct[i]))*i).astype(int) for i in idx_dct.keys()]).astype("uint8")
            mo_coeff = mo_coeff[:,idx]
            obj = OrbExtract(mol=mol,mo_coeff=mo_coeff,labels=labels)
            self.obj = obj
        else:
            self.obj = None

class DataWriter():
    def __init__(self,fn):
        if os.path.isfile(fn):
            os.system(f"rm {fn}")
        self.h5fn = fn
        self.onum = 0

    def write_orb(self,dct):
        with h5py.File(self.h5fn, "a") as f:
            for k,v in dct.items():
                if isinstance(v,dict):
                    for k2,v2 in v.items():
                        f.create_dataset(f"o{self.onum}/{k}_{k2}", data=v2)
                else:
                    f.create_dataset(f"o{self.onum}/{k}", data=v)
            self.onum += 1
        
import glob
from pyscf.tools import molden
names = list(glob.glob("../../../data/kl_constructed/constructed/*"))
h5fn = "../../data/kl_constructed.h5"
data_writer = DataWriter(h5fn)
for name in tqdm(names):
    tmobj = Tm_Obj(name)
    obj = tmobj.obj
    if obj:
        nmos = obj.mo_coeff.shape[1]
        for i in range(nmos):
            dct = tmobj.obj.extract_nlm(i)
            data_writer.write_orb(dct)