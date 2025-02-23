import os
import numpy as np
from deeporb.data_gen import OrbExtract
from deeporb.qh9 import QH9data
from tqdm import tqdm
import glob
import h5py

#You will need to install the datasets folder (with just the QH9Stable directory) from here:
#https://drive.google.com/drive/folders/1W2qb8Uu3CGwYMk1VX8vrHPVvA_cK8ARR
#Set NUM_MOLECULES to None to generate all data

NUM_MOLECULES = 1
edata = QH9data(root="../datasets")
if not NUM_MOLECULES:
    NUM_MOLECULES = len(edata.dataset)
    
fn = f"../data/qh9_{NUM_MOLECULES}_v12.h5"
if os.path.isfile(fn):
    os.system(f"rm {fn}")

onum = 0
for sys_idx in tqdm(range(NUM_MOLECULES)):
    mol, h, mo_energy, mos, mo_occ = edata.get(sys_idx)
    obj = OrbExtract(mol=mol, mo_ene=mo_energy, mo_coeff=mos, mo_occ=mo_occ)
    mo_idx = np.where((obj.mo_occ == 0)*(obj.mo_ene < 1.2))[0]
    for i in mo_idx:
        with h5py.File(fn, "a") as f:
            for k,v in obj.extract_nlm(i,rotate=False).items():
                if isinstance(v,dict):
                    for k2,v2 in v.items():
                        f.create_dataset(f"o{onum}/{k}_{k2}", data=v2)
                else:
                    f.create_dataset(f"o{onum}/{k}", data=v)
            onum += 1