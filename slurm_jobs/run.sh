#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --time=5-0
#SBATCH -o logs/train.out
#SBATCH -e logs/train.err
#SBATCH -a 0-15

uv run train.py --sub_idx $SLURM_ARRAY_TASK_ID --pretrain_loss mae --lr 0.0001 --num_epochs 1000 --d_model 64 --dim_feedforward 32 --patience 500 --use_rotary_encoding False --PATH_DATA /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/data --path_out /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/out_1000
