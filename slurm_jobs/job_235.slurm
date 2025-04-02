#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -o logs/train_log_235.out
#SBATCH -e logs/train_log_235.err

uv run train.py --sub_idx 235 --pretrain_loss mae --lr 0.0001 --num_epochs 100 \
    --d_model 16 --dim_feedforward 16 --patience 7 \
    --use_rotary_encoding False --apply_log_scaling False \
    --add_hour_to_features True \
    --PATH_DATA /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/data \
    --path_out /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/out_log/235
