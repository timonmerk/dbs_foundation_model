import os
import itertools

# Hyperparameter values
lr_list = ["0.1", "0.01", "0.001", "0.0001"]
d_model_list = [8, 16, 32, 64, 128, 256]
apply_log_scaling_list = [True, False]
dim_feedforward_list = [8, 16, 32, 64, 128, 256]

# Directory for SLURM scripts
os.makedirs("slurm_jobs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Base SLURM batch file template
base_script = """#!/bin/sh
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -o logs/train_log_{idx}.out
#SBATCH -e logs/train_log_{idx}.err

uv run train.py --sub_idx {idx} --pretrain_loss mae --lr {lr} --num_epochs 100 \\
    --d_model {d_model} --dim_feedforward {dim_feedforward} --patience 7 \\
    --use_rotary_encoding False --apply_log_scaling {log_scaling} \\
    --add_hour_to_features True \\
    --PATH_DATA /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/data \\
    --path_out /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/out_log/{idx}
"""

# Generate combinations and scripts
combinations = list(itertools.product(lr_list, d_model_list, apply_log_scaling_list, dim_feedforward_list))

for idx, (lr, d_model, log_scaling, dim_feedforward) in enumerate(combinations):
    script_content = base_script.format(
        idx=idx,
        lr=lr,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        log_scaling=str(log_scaling),
    )
    with open(f"slurm_jobs/job_{idx}.slurm", "w") as f:
        f.write(script_content)

print(f"Generated {len(combinations)} SLURM job scripts.")