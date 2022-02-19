#!/bin/bash

mkdir data
cd data
kaggle datasets download -d tejasreddy/iam-handwriting-top50
unzip iam-handwriting-top50.zip