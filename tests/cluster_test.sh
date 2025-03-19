#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --nodes=4
#SBATCH --tasks-per-node=2
#SBATCH --exclusive
#SBATCH --job-name=jax-sharding-test
#SBATCH --gpus=8
#SBATCH --qos=p71867_a40dual
#SBATCH --partition=zen2_0256_a40x2
##SBATCH --qos=p71867_a100dual
##SBATCH --partition=zen3_0512_a100x2
#SBATCH --output output/slurm-%j.out

nvidia-smi

source $DATA/venv-jax-sharding-test/bin/activate

cd ~/jax_array_info/

srun --output "tests/output/slurm-%2j-%2t.out" $DATA/venv-jax-sharding-test/bin/pytest tests/test_jax.py -vv -s
