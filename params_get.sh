#!/bin/bash

source .env/bin/activate

mkdir model_data
cd model_data

python ../handwritten-text-recognition-for-apache-mxnet/get_models.py
cd ..
