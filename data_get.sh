#!/usr/bin/env bash

mkdir data
cd data
kaggle datasets download -d naderabdalghani/iam-handwritten-forms-dataset
unzip iam-handwritten-forms-dataset.zip
rm iam-handwritten-forms-dataset.zip
