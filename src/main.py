from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow Framework
import tensorflow as tf
import tensorflow_datasets as tfds #Tensorflow dataset usage

# Basic Libraries for calculus and ploting
import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Data set inizialtiation with MNIST
dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Number class identification
number_class = [
    'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis',
    'Siete', 'Ocho', 'Nueve'
]
