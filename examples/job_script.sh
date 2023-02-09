#!/bin/bash

#SBATCH --job-name=SequentailMoreEpochs
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=20G

# source doesnt work.
#srun -u source activate /home/melissafm/anaconda3/envs/torchrl

# Datasets
#srun -u python sasrec_amazon.py --model='sasrec' --dataset='reviews_Books_5'
#srun -u python sasrec_amazon.py --model='sasrec' --dataset='reviews_Beauty_5'
#srun -u python sasrec_amazon.py --model='sasrec' --dataset='reviews_Electronics_5'

#srun -u python sasrec_amazon.py --model='ssept' --dataset='reviews_Books_5'
#srun -u python sasrec_amazon.py --model='ssept' --dataset='reviews_Beauty_5'
#srun -u python sasrec_amazon.py --model='ssept' --dataset='reviews_Electronics_5'

srun -u python sequential_recsys_amazondataset.py

echo "DONE"