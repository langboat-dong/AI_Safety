#!/bin/bash
#SBATCH -J project_5
#SBATCH -p JX-GPU-IB
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH --gres=gpu:2
#SBATCH -o train_resnet50.out

# python main_free.py DIR='./data'
python train_origin.py