#!/bin/bash
#SBATCH --job-name=slurm_train_exam # identifier for the job listings
#SBATCH --output=slurm_train_exam_main.log        # outputfile
#SBATCH --nodelist=3090-221
#SBATCH --gpus=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1                # every process wants one GPU!
##SBATCH --gpu-bind=none                   # NCCL can't deal with task-binding...
#SBATCH --gpus-per-node=8



# scontrol show hostnames

echo "$SLURM_JOB_NODELIST"
echo "$SLURM_NTASKS"

srun --output=slurm_train_exam_%j_%t.log nvidia-smi