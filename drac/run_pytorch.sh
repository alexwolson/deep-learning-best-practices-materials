#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --account=def-someone
#SBATCH --time=0-00:10:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=gpu-test-%j.out
#SBATCH --error=gpu-test-%j.err

module load python/3.10

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index torch

nvidia-smi

srun python gpu_test.py
