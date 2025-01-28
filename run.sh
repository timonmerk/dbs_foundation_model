#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --time=7-0
#SBATCH --cpus-per-task=256
#SBATCH -o logs/train.out
#SBATCH -e logs/train.err
uv run train.py 