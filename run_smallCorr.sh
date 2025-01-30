#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -o logs/small_corr.out
#SBATCH -e logs/small_corr.err
#SBATCH -a 0-15

uv run train.py --sub_idx $SLURM_ARRAY_TASK_ID --pretrain_loss mae --downstream_loss corr --downstream_label all --lr 0.0001 --num_epochs 100 --d_model 32 --dim_feedforward 8 --time_ar_head 4 --time_ar_layer 2 --patience 80 --use_rotary_encoding False --PATH_DATA /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/data --path_out /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/out_corr
