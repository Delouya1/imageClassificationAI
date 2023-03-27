import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt

(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data() # load the data
training_images, test_images = training_images / 255.0, test_images / 255.0  # normalize the data

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']  # the names of the classes

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

training_images = training_images[:20000]  # cut down the data to make training faster
training_labels = training_labels[:20000]
test_images = test_images[:4000]
test_labels = test_labels[:4000]

model = models.Sequential()  # create the model
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # add the first convolutional layer
model.add(layers.MaxPooling2D((2, 2)))  # add the first pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # add the second convolutional layer
model.add(layers.MaxPooling2D((2, 2)))  # add the second pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # add the third convolutional layer
model.add(layers.Flatten())  # flatten the output of the convolutional layers
model.add(layers.Dense(64, activation='relu'))  # add a dense layer
model.add(layers.Dense(10, activation='softmax'))  # add the output layer

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # compile the model

model.fit(training_images, training_labels, epochs=10, validation_data=(test_images, test_labels))  # train the model

loss, accuracy = model.evaluate(test_images, test_labels)  # evaluate the model
print("Loss: ", loss)
print("Accuracy: ", accuracy)

model.save('model.h5')  # save the mode
model = tf.keras.models.load_model('model.h5')  # load the model




