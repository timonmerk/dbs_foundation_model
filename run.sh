#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --time=5-0
#SBATCH -o logs/train.out
#SBATCH -e logs/train.err
#SBATCH -a 0-15

uv run train.py --sub_idx $SLURM_ARRAY_TASK_ID --lr 0.0001 --num_epochs 300 --d_model 64 --dim_feedforward 32 --patience 30 --use_rotary_encoding False --PATH_DATA /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/data --path_out /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/out