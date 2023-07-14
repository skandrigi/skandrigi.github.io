#App by Sandeep Kandrigi

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

#dataset
fashion_mnist = keras.datasets.fashion_mnist

#train images with 60k imgs, 10k for testing, load data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#matplotlib show all grayscale images
#plt.imshow(train_images[40000], cmap='gray', vmin=0, vmax = 255)
#plt.show()

#neural-net, input layer, hidden layer, output layer, sequential means vertical columns that connect to each other, like a linked list cant skip a sequential column o -> o -> o
#sequential command builds each layer of neural net and makes prediction based off of training in the output layer 
#build input layer first then output layer, then build hidden layers last based on necessity

model = keras.Sequential([
    #activation function is like a filtering function that stops or continues testing for a node/sorting group depending on value returned (ex: stop testing node if .9 or above)
    #dealing with 28x28 matrix image, input layer flattened to make it compatible to "pump" in all data, 28x28 flattened into 784x1 input layer, 1 pixel for each image
    keras.layers.Flatten(input_shape=(28, 28)),
    
    #hidden layer, Dense turns 784 into 128 nodes based on grouping, pick a number and play with it to get 128 or however many based on thoroughness and necessity for deep learning
    #relu gets rid of negative numbers and returns the value or 0
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    
    #output is 0-10 depending on piece of clothing, return maximum based on largest probability value, dense turn the 128 nodes into 10 based on probability
    #softmax find the largest probability
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])
#loss function to determine how far the predicted values deviate from the actual values in the training data
#Optimizers are the extended class, which include added information to train a specific model
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy')

#train model using training data, for each epoch is tests how wrong it is then optimizes, repeats 5 times in this case
model.fit(train_images, train_labels, epochs=5)

#test model, evaluating model for efficiency
test_loss = model.evaluate(test_images, test_labels)

plt.imshow(train_images[0], cmap='gray', vmin=0, vmax = 255)
plt.show()

print(test_labels[1])

#make predictions, return probability for matching each of the 10 pieces of clothing
predictions = model.predict(test_images)
print(predictions[0])

#print prediction
print(list(predictions[1]).index(max(predictions[1])))