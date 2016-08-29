#!/bin/bash

for learning_rate in 0.01 0.001 0.0001; do
    for kernel_size in 17 25 33; do
        for weight_decay in 0.0002 0.0; do
            sbatch imagenet.sh ${learning_rate} ${kernel_size} ${weight_decay}
        done
    done
done
