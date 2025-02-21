#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodelist=titan-222
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --output=debug_%j.log

echo $SLURM_NTASKS

# 각 태스크에 대해 hostname과 date 명령어 실행, 각각의 결과를 별도 파일에 저장
# srun --exclusive -n $SLURM_NTASKS sh -c 'hostname; date' > debug_${SLURM_JOB_ID}_$SLURM_TASK_PID.log

srun --exclusive -n $SLURM_NTASKS sh -c 'hostname; date' > debug_${SLURM_JOB_ID}_$SLURM_PROCID.log

NPROCS=$SLURM_NPROCS
echo NPROCS=$NPROCS > debug_${SLURM_JOB_ID}_summary.log

sleep 10