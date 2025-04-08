#!/bin/sh
#SBATCH --job-name=dl_transformer_567
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -a 0-15
#SBATCH -o logs/train_log_567_%a.out
#SBATCH -e logs/train_log_567_%a.err

uv run train.py --sub_idx $SLURM_ARRAY_TASK_ID --pretrain_loss mae --lr 0.0001 --num_epochs 100 \
    --d_model 256 --dim_feedforward 16 --patience 7 \
    --use_rotary_encoding False --apply_log_scaling False \
    --add_hour_to_features False \
    --PATH_DATA /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/data \
    --path_out /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/out_log/567
