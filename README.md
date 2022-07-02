# SELD_DCASE2022_Ric

This is my system for the DCASE 2022 task 3 challenge.

## Getting started

Setup the environment and install requirements:
```bash
conda env create -n dcase2022 --file requirements.yml
source activate dcase 2022

cd ./spaudiopy
pip install -e .

cd ./torch-audiomentations
pip install -e .
```

Note, spaudiopy can be installed from pip directly too
```bash
pip install spaudiopy
```
