"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+                                                                                                                      +
+   ITESM QRO. 2021 - Inteligent Systems                                                                               +
+   Name: Jose Carlos Pacheco Sanchez                                                                                  +
+   ID: A01702828                                                                                                      +
+                                                                                                                      +
+   Script Ob: Cleaner and direct use of funcitions for ploting and framework usage                                    +
+                                                                                                                      +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


numberNames = [
    'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six',
    'Seven', 'Eight', 'Nine'
]

# split the pixel numbers into black or wihte (255 to 0 or 1)
def imageWitheBlak(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


def plotImages(i, predictions, finalLabel):
    predictions, finalLabel = predictions[i], finalLabel[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions, color="#888888")
    plt.ylim([0, 1])
    finalLabels = np.argmax(predictions)

    thisplot[finalLabels].set_color('red')
    thisplot[finalLabel].set_color('blue')


def plotGraph(i, predictions, finalLabels, images):
    predictions, finalLabel, img = predictions[i], finalLabels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    finalLabels = np.argmax(predictions)
    if finalLabels == finalLabel:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("Prediccion: {}".format(numberNames[finalLabels]), color=color)