#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --time=7-0
#SBATCH --cpus-per-task=256
#SBATCH -o logs/train.out
#SBATCH -e logs/train.err
uv run train.py --lr 0.0001 --num_epochs 300 --d_model 64 --dim_feedforward 32 --patience 30 --use_rotary_encoding False --PATH_DATA /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/data --path_out /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/out