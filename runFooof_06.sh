#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --time=7-0
#SBATCH -o logs/fooof_06.out
#SBATCH -e logs/fooof_06.err
#SBATCH -a 0-15

uv run train.py --sub_idx $SLURM_ARRAY_TASK_ID --pretrain_fooof True --ap_loss_factor 0.6 --lr 0.0001 --num_epochs 50 --d_model 64 --dim_feedforward 32 --patience 50 --use_rotary_encoding False --PATH_DATA /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/data --path_out /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/out_fooof_06