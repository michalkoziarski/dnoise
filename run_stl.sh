#!/bin/bash

for lr in 0.01 0.001 0.0001; do
    for kernel in 5 9 13; do
        for wd in 0.0002 0.0; do
            sbatch stl.sh ${lr} ${kernel} ${wd}
        done
    done
done
