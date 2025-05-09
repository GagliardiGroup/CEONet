import os
import numpy as np
from deeporb.data_gen import OrbExtract
from tqdm import tqdm
import glob
import h5py

#Make sure to remove the moldens in the failed_sanity_check.pkl
#Set NUM_MOLECULES to None to generate all data

NUM_MOLECULES = 1
MOLDEN_DIR = "../../data/qm9_moldens"
names = glob.glob(f"{MOLDEN_DIR}/*.molden")
if not NUM_MOLECULES:
    NUM_MOLECULES = len(names)
    
fn = f"../data/sto3g_{NUM_MOLECULES}_occ.h5"
if os.path.isfile(fn):
    os.system(f"rm {fn}")

onum = 0
for name in tqdm(names[:NUM_MOLECULES]):
    obj = OrbExtract(name,rotate=False)
    mo_idx = np.where((obj.mo_occ == 2)*(obj.mo_ene > -1.75))[0]
    for i in mo_idx:
        with h5py.File(fn, "a") as f:
            for k,v in obj.extract_nlm(i,rotate=False).items():
                if isinstance(v,dict):
                    for k2,v2 in v.items():
                        f.create_dataset(f"o{onum}/{k}_{k2}", data=v2)
                else:
                    f.create_dataset(f"o{onum}/{k}", data=v)
            onum += 1