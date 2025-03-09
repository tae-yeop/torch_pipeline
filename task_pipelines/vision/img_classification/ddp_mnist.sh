#!/bin/bash -l

#SBATCH --job-name=ddp-mnist
#SBATCH --time=999:00:00
#SBATCH -p 80g
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8 
#SBATCH --mem=100G
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH -o ./logs/output_%A_%a.txt

export NCCL_SOCKET_IFNAME=eno

export TORCH_DISTRIBUTED_DEBUG=INFO # DETAIL
export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO 


export CONTAINER_IMAGE_PATH='/purestorage/project/tyk/0_Software/yolo_distil.sqsh'
export CACHE_FOL_PATH='/purestorage/project/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/home/tyk/torch_pipeline/vision/img_classification'


srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOL_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    python ddp_mnist.py \



# 그냥 torchrun 쓰는 방법
# torchrun --standalone --nnodes=1 --nproc_per_node=8 ddp_mnist.py