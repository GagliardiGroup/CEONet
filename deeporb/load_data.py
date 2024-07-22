import ase
import pandas as pd
import numpy as np
import torch
import h5py
from cace.cace.data.atomic_data import AtomicData
from cace.cace.tools import torch_geometric

def from_h5key(h5key,h5fn,cutoff=None):
    with h5py.File(h5fn, "r") as f:
        data = f[h5key]

        #Make atoms object
        pos = np.vstack([np.array(data[l]) for l in ["x","y","z"]]).T
        atoms = ase.Atoms(numbers=data["el"],positions=pos)
        ad = AtomicData.from_atoms(atoms,cutoff=cutoff) #makes graph structure
        ad.energy = torch.Tensor([data["ene"][()]])
        ad.occ = torch.Tensor(np.array(data["occ"])) #store as array, easier

        #Make ordata
        cols = list(data)
        names = []
        arrs = []
        for n in range(1,15):
            for l in range(3):
                if n - l > 12:
                    continue
                if l >= n:
                    continue
                arrs += [np.array(data[f"{n}_{l}"])]
        orbdata = np.vstack(arrs).T
        ad.orbdata = torch.Tensor(orbdata)
        return ad

def from_h5key_old(h5key,h5fn,cutoff=None):
    #Highly recommended to do preprocessing to cut down on cpu load
    
    with h5py.File(h5fn, "r") as f:
        data = f[h5key]

        #Make atoms object
        num = data["el"]
        occ = data["occ"]
        ene = data["ene"]
        pos = np.vstack([np.array(data[l]) for l in ["x","y","z"]]).T
        atoms = ase.Atoms(numbers=num,positions=pos)
        ad = AtomicData.from_atoms(atoms,cutoff=cutoff)
        ad.energy = torch.Tensor([ene[0]])
        ad.occ = torch.Tensor(np.array(occ)) #store as array, easier

        #Make ordata
        cols = list(data)
        names = []
        arrs = []
        #This step could be preprocessed:
        for n in range(1,15):
            for l in range(3):
                if n - l > 12:
                    continue
                if l >= n:
                    continue
                if l == 0:
                    arrs += [np.array(data[f"{n}_{l}_{0}"])]
                else:
                    sum = 0
                    for m in range(-l,l+1):
                        sum += np.array(data[f"{n}_{l}_{m}"])**2
                    arrs += [sum]
                # names += [f"{n}_{l}"]
        orbdata = np.vstack(arrs).T
        # ad.orbdata_names = names
        ad.orbdata = torch.Tensor(orbdata)
        return ad
