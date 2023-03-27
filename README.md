Image Classification with Neural Networks in Python

This project demonstrates how to use neural networks to classify images using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, 
with 6,000 images per class. T
he classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

Dependencies:
Python 3.x
TensorFlow 2.x
NumPy
OpenCV
Matplotlib
Installation

To install the required dependencies, run:

pip install tensorflow numpy opencv-python matplotlib

Usage

Load the CIFAR-10 dataset and normalize the data.
Define the class names.
Display a sample of the images with their corresponding labels.
Create a convolutional neural network (CNN) model.
Compile the model.
Train the model using the training data.
Evaluate the model using the test data.
Save the model to a file.
Load the model from the file.
Load an image to classify.
Preprocess the image.
Predict the class of the image.
Display the predicted class name.
Note: To speed up the training process, you can uncomment lines 15-18 and use a smaller subset of the data.

Example:
python image_classification.py

Output:
Prediction is: Horse

This means that the model correctly predicted that the image is of a horse.

Credits:

This project was created by Gil Delouya. The code is based on the TensorFlow tutorial Convolutional Neural Network (CNN). The CIFAR-10 dataset was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. More information about the dataset can be found at: https://www.cs.toronto.edu/~kriz/cifar.html.
