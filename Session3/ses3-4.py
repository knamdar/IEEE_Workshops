# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:10:27 2020

@author: Ernest Namdar
"""

import numpy as np
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Load the ONNX file
    model = onnx.load("./CNN.onnx")

    # Import the ONNX model to Tensorflow
    tf_rep = prepare(model)

    x_test[0].shape
    plt.imshow(x_test[0])
    inp = np.asarray(x_test[0], dtype=np.float32)[np.newaxis, np.newaxis, :, :]
    output = tf_rep.run(inp)
    print(output)
    print('The digit is classified as ', np.argmax(output))
