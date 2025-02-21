#!/bin/bash
#SBATCH --job-name=slurm_train_exam # identifier for the job listings
#SBATCH --output=check_multi_node_nccl.log        # outputfile
#SBATCH --nodelist=3090-221,titan-222
#SBATCH --gpus=16
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=8


##SBATCH --gpus-per-task=1                # every process wants one GPU!
#SBATCH --gpu-bind=none                   # NCCL can't deal with task-binding...

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


echo "$SLURM_JOB_NODELIST"
echo "$SLURM_NTASKS"
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

export NODE_RANK=$SLURM_NODENUM # $SLURM_PROCID, $SLURM_LOCALID

echo "NODE_RANK=${NODE_RANK}"



# srun --output=check_multi_node_nccl_%j_%t.log nvidia-smi

srun singularity exec --nv --nvccli -B /purestorage/:/purestorage /purestorage/slurm_test/pytorch_22.07-py3.sif \
torchrun \
--nnodes 2 \
--nproc_per_node 8 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint 172.100.100.10:29500 \
check_multi_node_train.py