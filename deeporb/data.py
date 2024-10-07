from cace.cace.tools.torch_geometric import Dataset, DataLoader #their loader to import dicts
from .load_data import from_h5key, from_h5cart
import lightning as L
import torch
import h5py
import os

class OrbDataset(Dataset):
    def __init__(self,root="data/aodata.h5",cutoff=4.0,cart=False,
                transform=None, pre_transform=None, pre_filter=None):
        self.cutoff = cutoff
        self.cart = cart
        super().__init__(root, transform, pre_transform, pre_filter)

    def len(self):
        with h5py.File(self.root, "r") as f:
            return len(f.keys())
    
    def get(self, idx):
        if self.cart:
            return from_h5cart(f"o{idx}",h5fn=self.root,cutoff=self.cutoff)
        else:
            return from_h5key(f"o{idx}",h5fn=self.root,cutoff=self.cutoff)

class OrbData(L.LightningDataModule):
    def __init__(self, num=None, root="data/aodata.h5", cutoff=4.0, cart=False,
                 batch_size=32, valid_p=0.2):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.valid_p = valid_p
        self.num = num
        self.cart = cart
        self.cutoff = cutoff
        self.prepare_data()
    
    def prepare_data(self):
        dataset = OrbDataset(self.root,cutoff=self.cutoff,cart=self.cart)
        if self.num is not None:
            dataset = dataset[:self.num]
        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        data_cutoff = int(len(dataset)*(1-self.valid_p))
        self.train = dataset[:data_cutoff]
        self.val = dataset[data_cutoff:]

    def train_dataloader(self):
        train_loader = DataLoader(self.train, batch_size=self.batch_size,
                                  shuffle=True, num_workers = os.cpu_count())
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val, batch_size=self.batch_size, 
                                shuffle=False, num_workers = os.cpu_count())
        return val_loader