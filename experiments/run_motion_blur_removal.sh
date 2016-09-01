#!/bin/bash

for learning_rate in 0.01; do
    for kernel_size in 17 25 33; do
        for weight_decay in 0.0002 0.0; do
            sbatch script.sh motion_blur_removal.py --learning_rate ${learning_rate} --kernel_size ${kernel_size} --weight_decay ${weight_decay}
        done
    done
done
