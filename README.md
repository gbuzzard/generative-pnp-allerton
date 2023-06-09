# generative-pnp-allerton
Code to implement methods and experiments for Generative Plug-and-Play: Posterior Sampling for Inverse Problems

If you use this code in your work, please cite

C.A. Bouman and G.T. Buzzard, "Generative Plug-and-Play: Posterior
Sampling for Inverse Problems," 2023 Allerton Conference on 
Communication, Control, and Computing (Allerton), 2023.  

Installation on MacOS and Linux:
--------------------------------
Assuming you have python and conda installed, then from a terminal open
in the generative-pnp-allerton directory, enter
```
cd dev_scripts
yes | source clean_install_all.sh
```
This will install and activate the conda enviroment gpnp.  

If you prefer, you can create a pip virtual environment.  From a terminal, change
directory to the generative-pnp-allerton directory, then enter 
```
yes | python -m venv gpnp-venv
source gpnp-venv/bin/activate
pip install -r requirements.txt
source gpnp-venv/bin/activate
```
This will create and activate the pip virtual environment gpnp-venv.

Experiments:
------------
After creating a virtual environment as above, the figures from the paper 
can be reproduced with the following procedure.

From a terminal, change to the `experiments` directory and activate the virtual 
environment using either `conda activate gpnp` or `source gpnp-venv/bin/activate`.

1. Butterfly example: `python gpnp_generation.py`  
2. CT example: `python gpnp_generation_svmbir.py`  

When finished, you can use `conda deactivate` (or `deactivate` for a pip virtual environment).  

