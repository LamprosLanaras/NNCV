#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00

# Pull container from dockerhub
apptainer pull container.sif docker://cclaess/5lsm0:v1

export APPTAINERENV_HF_TOKEN="MY_HUGGINGFACE_TOKEN"
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0

# Use the huggingface-cli package inside the container to download the data
mkdir -p data
apptainer exec container.sif \
    huggingface-cli download TimJaspersTue/5LSM0 --local-dir ./data --repo-type dataset

#--resume-download