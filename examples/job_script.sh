#!/bin/bash

#SBATCH --job-name=SASRecBeauty
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=20G

# source doesnt work.
#srun -u source activate /home/melissafm/anaconda3/envs/torchrl

# Datasets
srun -u python sasrec_amazon.py
#srun -u python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
#srun -u python main.py --dataset=Steam --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda

echo "DONE"