#!/bin/bash
# 여기선 노드 2개에서 기본적으로 gpu 8개 전부 다 쓸 때 해야하는거 확인
#SBATCH --job-name=slurm_train_exam # identifier for the job listings
#SBATCH --output=slurm_train_exam_main.log        # outputfile
#SBATCH --nodelist=3090-221,titan-222
#SBATCH --gpus=16
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=8


##SBATCH --gpus-per-task=1                # every process wants one GPU!
#SBATCH --gpu-bind=none                   # NCCL can't deal with task-binding...



scontrol show hostnames $SLURM_JOB_NODELIS

echo "$SLURM_JOB_NODELIST"
echo "$SLURM_NTASKS"

srun --output=slurm_train_exam_%j_%t.log nvidia-smi