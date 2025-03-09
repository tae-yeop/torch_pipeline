#!/bin/bash

#SBATCH --job-name=fr-adapter
#SBATCH --partition=80g
#SBATCH --nodelist=hpe161
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --gpus-per-task=8
#SBATCH --mem=512gb
#SBATCH --cpus-per-task=128
#SBATCH --output=adapter-%j.out

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

echo "Run started at:- "
date

srun singularity exec --nv -c -B /purestorage/:/purestorage /purestorage/project/tyk/slurm/imgs/hugging.sif \
bash -c "ln -s /purestorage/project/tyk/tmp ~/.cache && \
bash ./train.sh ./multi_gpu.yaml
"

echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"