#!/bin/bash
#SBATCH -n 48
#SBATCH -p fila3
WORKDIR=/home/grkremer/workspace/Dissertation

##wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
##bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/grkremer/miniconda3

export PATH="/home/grkremer/miniconda3/bin:$PATH"

conda env create -f environment.yml -n pymoo_test