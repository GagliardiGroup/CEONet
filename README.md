# CEONet
Cartesian equivariant deep learning for molecular orbitals 

## Overview
Thank you for your interest in this work!

Source code for CEONet is avialable in src/deeporb, particularly src/deeporb/ceonet.py

Training scripts are found in scripts/training_scripts

Notebooks show example inference and figure generation, using the data in model_eval

## Dataset Management

The code for generating model input as .h5 files can be found in src/deeporb/data_gen.py 

The key function "extract_modct" requires a PySCF mol object and molecular orbital coefficients, which can be read in from molden files or provided directly

The OrbData class in src/deeporb/data.py is then capable of reading the generated h5 files as model input

## Dependencies

CACE (https://github.com/dking072/cace, branch of https://github.com/BingqingCheng/cace)
ASE (https://wiki.fysik.dtu.dk/ase/install.html)
