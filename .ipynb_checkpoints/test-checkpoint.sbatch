#!/bin/bash
#SBATCH -n 48
#SBATCH -p fila3
WORKDIR=/home/grkremer/workspace/Dissertation


source /home/grkremer/miniconda3/bin/activate
conda env list
conda activate pymoo_test

python experiment.py