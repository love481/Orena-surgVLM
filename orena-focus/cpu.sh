#!/bin/bash
#SBATCH -A a0264
#SBATCH --job-name=data_process
#SBATCH --partition=xfer
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=/iopsstor/scratch/cscs/lpanta32/logs/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/lpanta32/logs/%x_%j.err

# ---- Setup ----
mkdir -p /iopsstor/scratch/cscs/lpanta32/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate surg_env

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# ---- Data processing command ----
python /capstor/store/cscs/swissai/a0264/lpanta32/orena-focus/create_data.py 

echo "Job finished at $(date)"