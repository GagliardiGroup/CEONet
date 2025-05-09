# CEONet
> Cartesian equivariant deep learning for molecular orbitals 

## Overview
Thank you for your interest in this work!

Source code for CEONet is avialable in src/deeporb, particularly src/deeporb/ceonet.py

Training scripts are found in scripts/training_scripts

Notebooks show example inference and figure generation, using the data in model_eval

## Installation

1. Clone the project
1. Make a directory in the project called `deeporb/opt`
1. Clone `cace` dependency from `https://github.com/dking072/cace`
1. Conda install the environment
1. Activate the environment

## Dataset Management

Running in-memory:

1. Convert `.h5` file to `.pt` file to store in memory.

`python scripts/memory-experiments.py --file /eagle/DeepOrb/sto3g/subset/sto3g_occ_100000.h5 --convert`

## Running on Polaris

*One time setup:*

Make sure that the `deeporb` environment is installed in the project direectory
1. Check the folder:`/eagle/DeepOrb/env/deeporb`
1. If not installed 

```
conda env create -f environment.yml —prefix=/eagle/DeepOrb/env/deeporb -n deeporb
```

1. Make sure you add its location to your personal `.condarc` file

```
conda config --append envs_dirs /eagle/DeepOrb/env
```

*Running training:*

```
module use /soft/modulefiles
module load conda
conda activate deeporb
cd /eagle/DeepOrb/deeporb
python scripts/polaris/sto3g_occ_1000000.py
```

*Interactive Job:*
`qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug -A DeepOrb`

## Dependencies

CACE (https://github.com/BingqingCheng/cace)
ASE (https://wiki.fysik.dtu.dk/ase/install.html)
