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

sbatch script.sh imagenet_classification.py -train_noise None -test_noise None

for noise in "${noises[@]}"
do
    sbatch script.sh imagenet_classification.py -train_noise "$noise" -test_noise "$noise"
    sbatch script.sh imagenet_classification.py -train_noise None -test_noise "$noise"
done
