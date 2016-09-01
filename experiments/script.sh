#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu

module add plgrid/tools/python/2.7.9
module add plgrid/apps/cuda/7.0

python ${1} ${@:2}
