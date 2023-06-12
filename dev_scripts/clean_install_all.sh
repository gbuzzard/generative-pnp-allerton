#!/bin/bash

# Create and activate new conda environment
# First check if the target environment is active and deactivate if so
NAME=gpnp
if [ "$CONDA_DEFAULT_ENV" == $NAME ]; then
    conda deactivate
fi

# Then remove the old version and reinstall
conda remove env --name $NAME --all
conda create --name $NAME python=3.8
conda activate $NAME
pip install -r ../requirements.txt

