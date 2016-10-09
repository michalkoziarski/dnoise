import sys

sys.path.append('..')

import models
import trainers
import loaders
import noise
import tensorflow as tf
import argparse


train_set, test_set = loaders.load_imagenet_labeled(batch_size=50, patch=256)
