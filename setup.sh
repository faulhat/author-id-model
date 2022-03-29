#!/usr/bin/env bash

source .env/bin/activate
pip install -r requirements.txt
pip uninstall tensorboard -y

echo "{
    \"port\": 8080,
    \"debug\": true
}" > config.json

# Setup for handwritten text segmentation submodule.

cd ..
git clone https://github.com/usnistgov/SCTK
cd SCTK
export CXXFLAGS="-std=c++11" && make config
make all
make check
make install
make doc
cd -

pip install pybind11 numpy setuptools
cd ..
git clone https://github.com/nmslib/hnswlib
cd hnswlib/python_bindings
python setup.py install
cd ../..
