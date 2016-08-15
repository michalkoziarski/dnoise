#!/bin/bash

for lr in 0.01 0.001 0.0001; do
    for kernel in 17 25 33; do
        sbatch imagenet.sh ${lr} ${kernel}
    done
done
