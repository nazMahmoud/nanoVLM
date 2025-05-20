#!/bin/bash

#SBATCH --job-name=train
#SBATCH --time=11:00:00
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # num of gpus
#SBATCH --gpus-per-task=l40s:1
#SBATCH --cpus-per-task=10 # use 10 per GPU
#SBATCH --mem=100G
#SBATCH --output=./output/%x-%j.out
#SBATCH --error=./output/%x-%j.err



module load anaconda/3
conda activate nanoVLM

COMMAND="python train.py"

echo "Running script"
echo "$COMMAND"
eval $COMMAND
echo "Done script"
