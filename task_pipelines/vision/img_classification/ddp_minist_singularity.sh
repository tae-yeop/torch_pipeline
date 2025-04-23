#!/bin/bash

#SBATCH --job-name=ddp_multi_node
#SBATCH --time=00:30:00
#SBATCH --nodelist=hpe159,hpe160
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=16
#SBATCH --gpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
##SBATCH --account=ai1

# -----------------------------------------------------------------------------
# 1) Slurm가 제공하는 환경 변수들
# -----------------------------------------------------------------------------
# $SLURM_JOB_NODELIST : 실제 노드 이름 리스트
# $SLURM_NNODES       : 전체 노드 개수
# $SLURM_NODEID       : 0..(nnodes-1)
# $SLURM_PROCID       : 전체 rank (0..ntasks-1)
# $SLURM_GPUS_PER_TASK: 위에서 --gpus-per-task로 지정한 GPU 수

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
master_addr=${nodes[0]}  # 첫 번째 노드를 rendezvous endpoint로 사용 # master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr


echo "MASTER_ADDR="$MASTER_ADDR
echo "[INFO] MASTER_ADDR=$MASTER_ADDR"
echo "[INFO] nodes array = ${nodes[@]}"
echo "[INFO] SLURM_NODEID=$SLURM_NODEID"
echo "[INFO] SLURM_PROCID=$SLURM_PROCID"

# -----------------------------------------------------------------------------
# 2) 환경 변수 설정
# -----------------------------------------------------------------------------
export NCCL_SOCKET_IFNAME=eno
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO # DETAIL
export TORCH_CPP_LOG_LEVEL=INFO 

export CONTAINER_IMAGE_PATH='/purestorage/project/tyk/0_Software/yolo_distil.sqsh'
export CACHE_FOL_PATH='/purestorage/project/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/home/tyk/torch_pipeline/vision/img_classification'

# -----------------------------------------------------------------------------
# 3) 노드/랭크 정보 설정
# -----------------------------------------------------------------------------
nnode=$SLURM_NNODES
node_rank=$SLURM_NODEID
ngpu_per_node=$SLURM_GPUS_PER_TASK
master_port=8882


srun singularity exec --nv --nvccli -B /purestorage/:/purestorage /purestorage/slurm_test/pytorch_22.07-py3.sif \
torchrun \
--nnodes 2 \
--nproc_per_node 8 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint 172.100.100.10:29500 \
/purestorage/slurm_test/7_multi-node-multi-gpu/train_ddp.py 5 exp1.yml
