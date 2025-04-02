#!/bin/sh
#SBATCH --job-name=dl_transformer_76
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -a 0-15
#SBATCH -o logs/train_log_76_%a.out
#SBATCH -e logs/train_log_76_%a.err

uv run train.py --sub_idx $SLURM_ARRAY_TASK_ID --pretrain_loss mae --lr 0.01 --num_epochs 100 \
    --d_model 8 --dim_feedforward 128 --patience 7 \
    --use_rotary_encoding False --apply_log_scaling True \
    --add_hour_to_features True \
    --PATH_DATA /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/data \
    --path_out /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/out_log/76
