import h5py
from tqdm import tqdm

"""
This does the preprocessing on the h5 to reduce the cpu load during training
Basically does the symmetrization of the projected orbital features by summing over the m components squared
Simlar to eq. 4 in the Cace paper
"""

filename = "data/aodata.h5" #original h5
filename_2 = "data/aodata2.h5" #new h5

def rewrite_data(filename="data/aodata.h5",filename_2="data/aodata2.h5"):
    with h5py.File(filename, "r") as f1:
        h5keys = list(f1.keys())
    for h5key in tqdm(h5keys):
        with h5py.File(filename, "r") as f1:
            data = f1[h5key]
            to_copy_int = ["el","occ"]
            to_copy_float = ["x","y","z"]
            with h5py.File(filename_2, "a") as f2:
                if h5key in f2.keys():
                    continue
                for col in to_copy_int:
                    f2.create_dataset(f"{h5key}/{col}", data=data[col], dtype='uint8')
                for col in to_copy_float:
                    f2.create_dataset(f"{h5key}/{col}", data=data[col], dtype='float16')
                f2.create_dataset(f"{h5key}/ene", data=data["ene"][0], dtype='float16')
        
            #nl features 
            for n in range(1,15):
                for l in range(3):
                    if n - l > 12:
                        continue
                    if l >= n:
                        continue
                    if l == 0:
                        arr = np.array(data[f"{n}_{l}_{0}"])
                    else:
                        sum = 0
                        for m in range(-l,l+1):
                            sum += np.array(data[f"{n}_{l}_{m}"])**2
                        arr = sum
                    with h5py.File(filename_2, "a") as f2:
                        f2.create_dataset(f"{h5key}/{n}_{l}", data=arr, dtype='float16')

rewrite_data()