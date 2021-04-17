"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+                                                                                                                      +
+   ITESM QRO. 2021 - Inteligent Systems                                                                               +
+   Name: Jose Carlos Pacheco Sanchez                                                                                  +
+   ID: A01702828                                                                                                      +
+                                                                                                                      +
+   Script Ob: Main usage of the Script and variant desition on epoc and block nunbers learning                        +
+                                                                                                                      +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and KERAS Framework
import tensorflow_datasets as tfds
from libs import *

# Basic Libraries
import logging
import matplotlib.pyplot as plt
import math
import numpy as np

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
datasetTesting, datasetTest = dataset['train'], dataset['test']

# Initializaction of test numbers
number_train = metadata.splits['train'].num_examples
number_examples = metadata.splits['test'].num_examples
datasetTesting = datasetTesting.map(imageWitheBlak)
datasetTest = datasetTest.map(imageWitheBlak)

# General initiation of the neural Network with the number of layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # Clasiffication
])

# Compile function to use
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Learning rate into 32 blocks
size = 32
datasetTesting = datasetTesting.repeat().shuffle(number_train).batch(size)
datasetTest = datasetTest.batch(size)

# Start the Model learning process
model.fit(
    datasetTesting, epochs=1,  # Number of epocs to use
    steps_per_epoch=math.ceil(number_train / size)
)

# Evaluation of the Model
loss, accuracy = model.evaluate(
    datasetTest, steps=math.ceil(number_examples / 32)
)

print("Test Results!!: ", accuracy)

# From the dataset Image we take each one and testit on the labels

for images, labeles in datasetTest.take(1):
    images = images.numpy()
    labeles = labeles.numpy()
    predictions = model.predict(images)

rows = 5
cols = 3
total = rows * cols

plt.figure(figsize=(2 * 2 * cols, 2 * rows))
for i in range(total):
    plt.subplot(rows, 2 * cols, 2 * i + 1)
    plotGraph(i, predictions, labeles, images)
    plt.subplot(rows, 2 * cols, 2 * i + 2)
    plotImages(i, predictions, labeles)

plt.show()
