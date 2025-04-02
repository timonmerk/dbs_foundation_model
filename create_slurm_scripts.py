import os
import itertools

# Define parameter options
lr_list = ["0.1", "0.01", "0.001", "0.0001"]
d_model_list = [8, 16, 32, 64, 128, 256]
apply_log_scaling_list = [True, False]
dim_feedforward_list = [8, 16, 32, 64, 128, 256]

# Output directories
os.makedirs("slurm_jobs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# SLURM script header with array!
slurm_header = """#!/bin/sh
#SBATCH --job-name=dl_transformer_{idx}
#SBATCH --mem=100GB
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH -a 0-15
#SBATCH -o logs/train_log_{idx}_%a.out
#SBATCH -e logs/train_log_{idx}_%a.err
"""

# Command using $SLURM_ARRAY_TASK_ID as sub_idx
command_template = """
uv run train.py --sub_idx $SLURM_ARRAY_TASK_ID --pretrain_loss mae --lr {lr} --num_epochs 100 \\
    --d_model {d_model} --dim_feedforward {dim_feedforward} --patience 7 \\
    --use_rotary_encoding False --apply_log_scaling {log_scaling} \\
    --add_hour_to_features True \\
    --PATH_DATA /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/data \\
    --path_out /data/cephfs-1/home/users/merkt_c/work/dbs_foundation_model/out_log/{idx}
"""

# Generate all combinations
combinations = list(itertools.product(lr_list, d_model_list, apply_log_scaling_list, dim_feedforward_list))

# Write scripts
for idx, (lr, d_model, log_scaling, dim_feedforward) in enumerate(combinations):
    slurm_script = slurm_header.format(idx=idx) + command_template.format(
        idx=idx,
        lr=lr,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        log_scaling=str(log_scaling),
    )
    with open(f"slurm_jobs/job_{idx}.slurm", "w") as f:
        f.write(slurm_script)

print(f"✅ Generated {len(combinations)} SLURM job array scripts (each with 16 tasks).")