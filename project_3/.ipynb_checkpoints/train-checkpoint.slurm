#!/bin/bash
#SBATCH -J project_3
#SBATCH -p JX-GPU-IB
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH --gres=gpu:1
#SBATCH -o project_3.out

python main.py
