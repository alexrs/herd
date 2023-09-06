#!/bin/sh
#BSUB -q gpua100
#BSUB -J fine-tune-llama
#BSUB -W 23:00
#BSUB -B
#BSUB -N
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
python herd/herd.py finetune_experts
