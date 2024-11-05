import numpy as np
from deeporb.data_gen import OrbExtract
from tqdm import tqdm
import glob
import h5py

#Make sure to remove the moldens in the failed_sanity_check.pkl!

#Rough means and standard deviations (mean,std)
#virt --> 0.6872, 0.1880
#occ --> -0.6637, 0.2863

#Make valence occupied
fn = "../data/aocart_occ.h5"
MOLDEN_DIR = "../../data/qm9_moldens" #dir where the moldens are
names = glob.glob(f"{MOLDEN_DIR}/*.molden")
onum = 0
for name in tqdm(names[:5000]): #change here to make more or less
    obj = OrbExtract(name,rotate=False)
    mo_idx = np.where((obj.mo_ene > -8)*(obj.mo_occ == 2))[0]
    for i in mo_idx:
        with h5py.File(fn, "a") as f:
            for k,v in obj.extract_nlm(i,rotate=False).items():
                if isinstance(v,dict):
                    for k2,v2 in v.items():
                        f.create_dataset(f"o{onum}/{k}_{k2}", data=v2)
                else:
                    f.create_dataset(f"o{onum}/{k}", data=v)
            onum += 1