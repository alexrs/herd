#!/bin/sh
#BSUB -q gpua100
#BSUB -W 23:00
#BSUB -B
#BSUB -N
#BSUB -J test_activations
### request the number of GPUs
#BSUB -gpu "num=1::mode=exclusive_process"
### request the number of CPU cores (at least 4x the number of GPUs)
#BSUB -n 4
### we want to have this on a single node
#BSUB -R "span[hosts=1]"
### we need to request CPU memory, too (note: this is per CPU core)
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load cuda/11.6
module load python3/3.11.4

source .venv/bin/activate

export HF_DATASETS_CACHE="/work3/s212722/herd/cache"

source .env

# set the wandb project where this run will be logged
export WANDB_PROJECT="herd-llama"
# turn off watch to log faster
export WANDB_WATCH="false"

python main.py test_molora --config-file=config/config_alpaca_10_q_v_r4.ini