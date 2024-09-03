#!/bin/bash
conda install pytorch=1.8 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge torchdiffeq
pip install pandas h5py ml-collections matplotlib scikit-learn