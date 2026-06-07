#!/bin/bash
#SBATCH --job-name=transfer_cv5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cv5_%j.out
#SBATCH --error=logs/cv5_%j.err

mkdir -p logs

echo "=== Job info ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "================"

cd /gpfs/users/rahhouis/Digital-Twin-Fault-Diagnosis/scripts

/gpfs/users/rahhouis/.conda/envs/projet/bin/python train_transfer_learning_cv5.py

echo "=== Done at $(date) ==="
