#!/bin/sh
script_name=$1
script_path=$2
job_id=${3:-}

for i in $(seq 0 9); do
    if [ -z "$job_id" ]
    then
        bsub \
        -q "gpua100" \
        -W "23:00" \
        -B \
        -N \
        -J "${script_name}_${i}" \
        -gpu "num=1::mode=exclusive_process" \
        -n "4" \
        -R "span[hosts=1]" \
        -o logs/%J.out \
        -e logs/%J.err \
        -R "rusage[mem=8GB]" "${script_path} expert_${i}"
    else
        bsub \
        -q "gpua100" \
        -W "23:00" \
        -B \
        -N \
        -J "${script_name}_${i}" \
        -gpu "num=1::mode=exclusive_process" \
        -n "4" \
        -R "span[hosts=1]" \
        -R "rusage[mem=8GB]" \
        -o logs/%J.out \
        -e logs/%J.err \
        -w "done(${job_id})" "${script_path} expert_${i}"
    fi
done
