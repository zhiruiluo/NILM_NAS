#!/bin/bash

#SBATCH --partition=backfill
#SBATCH --job-name=cuda_version
#SBATCH --output=logging/cuda_version.log


### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --exclude=discovery-g[1]

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2

#SBATCH --time=3-24:00:00

# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
# conda activate {{CONDA_ENV}}

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=($nodes)

worker_num=$(($SLURM_JOB_NUM_NODES)) #number of nodes other than the head node
for ((i = 0; i <= $worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w $node_i nvcc --version &
    srun --nodes=1 --ntasks=1 -w $node_i nvidia-smi &
    echo ""
    sleep 1
done