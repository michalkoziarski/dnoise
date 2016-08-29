#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu

module add plgrid/tools/python/2.7.9
module add plgrid/apps/cuda/7.0

python imagenet_motion_blur.py -learning_rate=$1 -kernel_size=$2 -weight_decay=$3
