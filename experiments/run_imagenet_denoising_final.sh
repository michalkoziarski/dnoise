#!/bin/bash

noises=(
    'GaussianNoise(0.05)'
    'GaussianNoise(0.10)'
    'GaussianNoise(0.20)'
    'GaussianNoise(0.50)'
    'QuantizationNoise(0.05)'
    'QuantizationNoise(0.10)'
    'QuantizationNoise(0.20)'
    'QuantizationNoise(0.50)'
    'SaltAndPepperNoise(0.05)'
    'SaltAndPepperNoise(0.10)'
    'SaltAndPepperNoise(0.20)'
    'SaltAndPepperNoise(0.50)'
    'RandomNoise(GaussianNoise)'
    'RandomNoise(QuantizationNoise)'
    'RandomNoise(SaltAndPepperNoise)'
    'RandomNoise()'
)

for noise in "${noises[@]}"
do
    sbatch script.sh imagenet_denoising_final.py -noise "$noise"
done
