import click
import time
import h5py
from pathlib import Path
import torch
from cace.data.atomic_data import AtomicData
from cace.tools.torch_geometric import Dataset, DataLoader
from deeporb.data import OrbData

def print_structure(name, obj):
    print(name)

def inspect_h5(file_path:Path):
    with h5py.File(file_path, 'r') as f:
        f.visititems(print_structure)

def load_h5_as_dict(file_path:Path):
    data_dict = {}
    with h5py.File(file_path, 'r') as h5_file:
        for group_name in h5_file.keys():
            group = h5_file[group_name]
            group_data = {}
            for key in group.keys():
                group_data[key] = group[key][()]
            data_dict[group_name] = group_data
    return data_dict

@click.command()
@click.option("--file", "-f", required=True, type=str)
@click.option("--convert", "-c", is_flag=True, default=0)
def main(file, convert):
    p = Path(file) 
    if convert:
        data = load_h5_as_dict(p)
        for group, datasets in data.items():
            for key, value in datasets.items():
                if not isinstance(value, torch.Tensor):
                    datasets[key] = torch.tensor(value)
        print("Converting h5 to pt")
        torch.save(data, p.with_suffix(".pt"))
    else:
        print("Loading file into dataset")
        start = time.time()
        ds = OrbData(p, in_memory=True)
        end = time.time()
        print(end - start)


if __name__ == "__main__":
    main()
