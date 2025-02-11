#!/bin/bash
# THIS FILE IS GENERATED BY AUTOMATION SCRIPT! PLEASE REFER TO ORIGINAL SCRIPT!
# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO PRODUCTION!

#SBATCH --partition={{PARTITION_NAME}}
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output={{JOB_DIR}}/{{JOB_NAME}}.log
{{GIVEN_NODE}}

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes={{NUM_NODES}}
#SBATCH --exclusive
#SBATCH --exclude=discovery-g[1]

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task={{NUM_GPUS_PER_NODE}}

#SBATCH --time=3-24:00:00

# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
# conda activate {{CONDA_ENV}}
{{LOAD_ENV}}

################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
# redis_password=$(uuidgen)
# export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=($nodes)

node_head=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w $node_head hostname --ip-address) # making redis-address
# node_head="discovery-l2.cluster.local"
# ip="192.168.211.14"


if [[ $ip == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$ip"
  if [[ ${#ADDR[0]} > 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "We detect space in ip! You are using IPV6 address. We split the IPV4 address as $ip"
fi

port=8786
dashboard_port=8787
nanny_port=8789
ip_head=$ip:$port
export ip_head
head_node_ip=$ip
export head_node_ip
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "STARTING HEAD at $node_head"
# srun --nodes=1 --ntasks=1 -w $node_head start-head.sh $ip $redis_password &
# RAY_worker_register_timeout_seconds=120
# export RAY_worker_register_timeout_seconds
srun --nodes=1 --ntasks=1 -w $node_head  \
  dask scheduler --host=$ip --port=$port --dashboard-address=$dashboard_port --idle-timeout=120 &
  # ray start --head --node-ip-address=$ip --port=$port --block --log-color=false --dashboard-host="0.0.0.0" &
sleep 10

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= $worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i \
    dask worker tcp://$ip_head --death-timeout=120 --dashboard-address=$dashboard_port --nanny-port=$nanny_port & 
    # ray start --address $ip_head --block --log-color=false &
  sleep 10
done

##############################################################################################

#### call your code below
{{COMMAND_PLACEHOLDER}} {{COMMAND_SUFFIX}}
