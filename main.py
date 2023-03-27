import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt

(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
training_images, test_images = training_images / 255.0, test_images / 255.0

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

