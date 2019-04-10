import sys
import os
import numpy as np

# reading training data
training_en = open('./training/hansards.36.2.e').read().splitlines()
training_fr = open('./training/hansards.36.2.f').read().splitlines()

# reading validation data
validation_en = open('./validation/dev.e').read().splitlines()
validation_fr = open('./validation/dev.f').read().splitlines()

# print(len(training_en), len(training_fr), len(validation_en), len(validation_fr))

k = 0
