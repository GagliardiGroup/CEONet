import ase
import os
import numpy as np
import torch
import h5py
import lightning as L
from pathlib import Path
from cace.data.atomic_data import AtomicData
from cace.tools.torch_geometric import Dataset, DataLoader

def from_h5key(h5key,h5fn,cutoff=None,avge0=0,sigma=1):
    with h5py.File(h5fn, "r") as f:
        data = dict(f[h5key])
        return process_mo_dictionary(data, cutoff, avge0, sigma)

def process_mo_dictionary(data:dict, cutoff, avge0, sigma):
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
    if "is_homo" in data.keys():
        if np.array(data["is_homo"]):
            l = 1
        elif np.array(data["is_lumo"]):
            l = 2
        else:
            l = 0
        ad.hl_label = l

    if "labels" in data.keys():
        ad.label = torch.Tensor(data["labels"][()]).to(torch.int64)
    if "occ" in data.keys():
        ad.occ = torch.Tensor((np.array(data["occ"])/2)).float() #float for BCE loss, go to 0/1
    if "energy" in data.keys():
        ad.energy = torch.Tensor(np.array(data["energy"]))
        ad.energy_ssh = 1/sigma * (ad.energy - avge0)
    # if "charge" in data.keys():
    #     ad.charge = torch.from_numpy(np.ones_like(els) * np.array(data["charge"])).int()
    return ad

class OrbDataset(Dataset):
    def __init__(self,root="data/aodata.h5",cutoff=7.6, avge0=0, sigma=1,
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
    def __init__(self,root="data/aocart.h5",cutoff=7.6, inmem_parallel=False,
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

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]

# class OrbData(L.LightningDataModule):
#     def __init__(self, root="data/aodata.h5", cutoff=4.0, in_memory=False, inmem_parallel=False, drop_last=True,
#                  batch_size=32, num_train=None, num_val=None, valid_p=0.1, test_p=0.1, avge0=0, sigma=1):
#         super().__init__()
#         self.batch_size = batch_size
#         self.root = root
#         self.valid_p = valid_p
#         self.test_p = test_p
#         self.cutoff = cutoff
#         self.avge0 = avge0
#         self.sigma = sigma
#         self.in_memory = in_memory
#         self.num_train = num_train
#         self.num_val = num_val
#         self.drop_last = drop_last
#         self.inmem_parallel = inmem_parallel
#         try:
#             self.num_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
#         except:
#             self.num_cpus = os.cpu_count()
#
#         self.prepare_data()
#
#     def prepare_data(self):
#         if not self.in_memory:
#             print("Setting up dataset")
#             dataset = OrbDataset(self.root,cutoff=self.cutoff,avge0=self.avge0,sigma=self.sigma)
#         else:
#             print("Loading dataset into memory")
#             dataset = OrbInMemoryDataset(self.root,cutoff=self.cutoff,avge0=self.avge0,
#                                          sigma=self.sigma,inmem_parallel=self.inmem_parallel)
#         torch.manual_seed(12345)
#         dataset = dataset.shuffle()
#         cut1 = int(len(dataset)*(1-self.valid_p-self.test_p))
#         cut2 = int(len(dataset)*(1-self.test_p))
#         self.train = dataset[:cut1]
#         if self.num_train:
#             self.train = self.train[:self.num_train]
#             assert(self.num_train == len(self.train))
#         self.val = dataset[cut1:cut2]
#         if self.num_val:
#             self.val = self.val[:self.num_val]
#             assert(self.num_val == len(self.val))
#         self.test = dataset[cut2:]
#
#     def train_dataloader(self):
#         train_loader = DataLoader(self.train, batch_size=self.batch_size, drop_last=self.drop_last,
#                                   shuffle=True, num_workers = self.num_cpus)
#         return train_loader
#
#     def val_dataloader(self):
#         val_loader = DataLoader(self.val, batch_size=self.batch_size, drop_last=False,
#                                 shuffle=False, num_workers = self.num_cpus)
#         return val_loader
#
#     def test_dataloader(self):
#         test_loader = DataLoader(self.test, batch_size=self.batch_size, drop_last=False,
#                                 shuffle=False, num_workers = self.num_cpus)
#         return test_loader
#
#

class SimpleOrbDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class OrbData(L.LightningDataModule):
    def __init__(self,
                 data_path="data/aodata.h5",
                 batch_size=32,
                 train_split=0.8,
                 val_split=0.1,
                 test_split=0.1,
                 seed=42,
                 cutoff=4.0,
                 avge0=0,
                 sigma=1
        ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.cutoff = cutoff
        self.avge0 = avge0
        self.sigma = sigma
        self.data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None


    def get_h5(self):
        return from_h5key(f"o{i}",h5fn=self.data_path,cutoff=self.cutoff,avge0=self.avge0,sigma=self.sigma)


    #def prepare_data(self):
    #    print("calling prepare data")
    #    if not self.data:
    #        p = Path(self.data_path)
    #        if p.suffix == '.h5':
    #            print("reading h5 file")
    #            with h5py.File(self.root, "r") as f:
    #                data_len = len(f.keys())
    #                self.data = [self.get_h5(i) for i in range(data_len)]
    #        elif p.suffix == '.pt':
    #            print("reading pt file")
    #            self.data = [process_mo_dictionary(v, self.cutoff, self.avge0, self.sigma) for k,v in torch.load(p).items()]
    #            print(len(self.data))

    def setup(self, stage=None):
        print("calling prepare data")
        if not self.data:
            p = Path(self.data_path)
            if p.suffix == '.h5':
                print("reading h5 file")
                with h5py.File(self.root, "r") as f:
                    data_len = len(f.keys())
                    self.data = [self.get_h5(i) for i in range(data_len)]
            elif p.suffix == '.pt':
                print("reading pt file")
                self.data = [process_mo_dictionary(v, self.cutoff, self.avge0, self.sigma) for k,v in torch.load(p).items()]
                print(len(self.data))
        print("calling setup")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        total_len = len(self.data)
        indices = np.arange(total_len)
        np.random.shuffle(indices)
        train_end = int(total_len * self.train_split)
        val_end = train_end + int(total_len * self.val_split)
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        train_data = [self.data[i] for i in train_idx]
        val_data = [self.data[i] for i in val_idx]
        test_data = [self.data[i] for i in test_idx]

        print(len(self.data))
        print(len(train_data))
        self.train_dataset = SimpleOrbDataset(train_data)
        self.val_dataset = SimpleOrbDataset(val_data)
        self.test_dataset = SimpleOrbDataset(test_data)


    def get_example_batch(self):
        if not self.data:
            self.prepare_data()
            temp_data = self.data[:self.batch_size]
            temp_dataset = SimpleOrbDataset(temp_data)
            temp_loader = DataLoader(temp_dataset, batch_size=self.batch_size)
            example_batch = next(iter(temp_loader))
            return example_batch


    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=15)
        return train_loader


    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=15)
        return val_loader


    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=15)
        return test_loader
