#!/bin/bash

# Create and activate new conda environment
# First check if the target environment is active and deactivate if so
NAME=gpnp

ENV_STRING=$((conda env list) | grep $NAME)
if [[ $ENV_STRING == *$NAME* ]]; then
  echo conda deactivate
fi
cd ..

# Then remove the old version and reinstall
conda remove env --name gpnp --all
conda create --name gpnp python=3.8
conda activate gpnp
pip install -r requirements.txt

cd dev_scripts

