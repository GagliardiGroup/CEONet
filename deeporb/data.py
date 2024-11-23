import ase
import os
import numpy as np
import torch
import h5py
import lightning as L

#loader to import dicts:
from cace.data.atomic_data import AtomicData
from cace.tools.torch_geometric import Dataset, DataLoader

def from_h5key(h5key,h5fn,cutoff=None,avge0=0,sigma=1):
    with h5py.File(h5fn, "r") as f:
        data = f[h5key]

        #Make atoms object
        els = np.array(data["atomic_numbers"])
        pos = np.array(data["positions"])
        atoms = ase.Atoms(numbers=els,positions=pos)
        ad = AtomicData.from_atoms(atoms,cutoff=cutoff) #makes graph structure

        #Orbdata
        #lao x (dim + 3) , 3 = alpha w atom_num
        #need to add pointer for orbdata lol
        ad["c"] = torch.from_numpy(np.array(data["c"]))
        ad["c_ptr"] = torch.tensor([ad["c"].shape[0]]).int()
        for l in range(3):
            if f"orbints_{l}" in data.keys():
                ad[f"orbints_{l}"] = torch.from_numpy(np.array(data[f"orbints_{l}"])).int()
                ad[f"orbdata_{l}_ptr"] = torch.tensor([ad[f"orbints_{l}"].shape[0]]).int()
                ad[f"orbfloats_{l}"] = torch.from_numpy(np.array(data[f"orbfloats_{l}"]))

        #Labels
        # occ = np.ones_like(els) * np.array(data["occ"])
        # ad.occ_atom = torch.from_numpy(occ)
        ad.energy = torch.Tensor(np.array(data["energy"]))
        ad.energy_ssh = 1/sigma * (ad.energy - avge0)
        if "charge" in data.keys():
            ad.charge = torch.from_numpy(np.ones_like(els) * np.array(data["charge"])).int()
        
        return ad

class OrbDataset(Dataset):
    def __init__(self,root="data/aodata.h5",cutoff=4.0, avge0=0, sigma=1,
                transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.cutoff = cutoff
        self.avge0 = avge0
        self.sigma = sigma

    def len(self):
        with h5py.File(self.root, "r") as f:
            return len(f.keys())
    
    def get(self, idx):
        return from_h5key(f"o{idx}",h5fn=self.root,cutoff=self.cutoff,avge0=self.avge0,sigma=self.sigma)

class OrbInMemoryDataset(Dataset):
    def __init__(self,root="data/aocart.h5",cutoff=4.0, inmem_parallel=False,
                transform=None, pre_transform=None, pre_filter=None, avge0=0, sigma=1):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.cutoff = cutoff
        self.avge0 = avge0
        self.sigma = sigma
        self.inmem_parallel = inmem_parallel
        self.prepare_data()

    def get_h5(self,i):
        return from_h5key(f"o{i}",h5fn=self.root,cutoff=self.cutoff,avge0=self.avge0,sigma=self.sigma)
    
    def prepare_data(self):
        #Currently takes ~5min to load small dataset serially
        with h5py.File(self.root, "r") as f:
            data_len = len(f.keys())
        if not self.inmem_parallel:
            self.dataset = [self.get_h5(i) for i in range(data_len)]
        else:
            #For some reason this breaks, maybe because of vm pool limit?
            #Worth trying on cluster though
            pool = torch.multiprocessing.Pool(processes=torch.multiprocessing.cpu_count())
            self.dataset = pool.map(self.get_h5, range(data_len))

    def len(self):
        return len(self.dataset)
    
    def get(self, idx):
        return self.dataset[idx]

class OrbData(L.LightningDataModule):
    def __init__(self, root="data/aodata.h5", cutoff=4.0, in_memory=False, inmem_parallel=False, drop_last=True,
                 batch_size=32, num_train=None, valid_p=0.1, test_p=0.1, avge0=0, sigma=1):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.valid_p = valid_p
        self.test_p = test_p
        self.cutoff = cutoff
        self.avge0 = avge0
        self.sigma = sigma
        self.in_memory = in_memory
        self.num_train = num_train
        self.drop_last = drop_last
        self.inmem_parallel = inmem_parallel
        try:
            self.num_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        except:
            self.num_cpus = os.cpu_count()
        
        self.prepare_data()
    
    def prepare_data(self):
        if not self.in_memory:
            dataset = OrbDataset(self.root,cutoff=self.cutoff,avge0=self.avge0,sigma=self.sigma)
        else:
            dataset = OrbInMemoryDataset(self.root,cutoff=self.cutoff,avge0=self.avge0,
                                         sigma=self.sigma,inmem_parallel=self.inmem_parallel)
        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        cut1 = int(len(dataset)*(1-self.valid_p-self.test_p))
        cut2 = int(len(dataset)*(1-self.test_p))
        self.train = dataset[:cut1]
        self.val = dataset[cut1:cut2]
        self.test = dataset[cut2:]
        
    def train_dataloader(self):
        train_loader = DataLoader(self.train, batch_size=self.batch_size, drop_last=self.drop_last,
                                  shuffle=True, num_workers = self.num_cpus)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val, batch_size=self.batch_size, drop_last=False,
                                shuffle=False, num_workers = self.num_cpus)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test, batch_size=self.batch_size, drop_last=False,
                                shuffle=False, num_workers = self.num_cpus)
        return test_loader