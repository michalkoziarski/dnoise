#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --time=72:00:00
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu

module add plgrid/tools/python/2.7.9
module add plgrid/apps/cuda/7.0

python stl_motion_blur.py -lr=$1 -kernel=$2 -wd=$3
