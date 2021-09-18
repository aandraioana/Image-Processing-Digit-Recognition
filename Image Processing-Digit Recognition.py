# -*- coding: utf-8 -*-

# -- Sheet --

# # **Introduction**
# The MNIST dataset is a large database of handwritten digits. It commonly used for training various image processing systems. 
# 
# MNIST is short for Modified National Institute of Standards and Technology database.This dataset is used for training models to recognize handwritten digits. This has an application in scanning for handwritten pin-codes on letters.
# 
# MNIST contains a collection of **70,000**, **28 x 28** images of handwritten digits from **0 to 9**.


from keras.datasets import mnist
from matplotlib import pyplot
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
#load the dataset and check the info about it
df = mnist.load_data()
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(y_test.shape))

# # **Dataset Example**


import matplotlib.pyplot as plt


print(y_train[600]) # An example
plt.imshow(X_train[600], cmap='Greys')

# # Reshaping and Normalizing the Images
# The dimension of the training data is (60000, 28, 28). In order to work with Keras we need it to be 4-dims NumPy arrays, so we reshape it with the following code:



X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# We need the data to be numerical so we convert it to float, also we must normalize it before we train a Neaural Network on it. Our values range between 0 and 255, as they represent RBG codes, so we divide the RGB codes to 255. 


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# after the normalization the range will be (0,1)
X_train =X_train/ 255
X_test =X_test/ 255


# # Building the Neural Network Model
# Deep learning models perform better with more data, however, they also take longer to train, especially when they start becoming more complex.


# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=input_shape))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
# Add the third hidden layer
model.add(Dense(50, activation='relu'))
# Add the output layer
model.add(Dense(10, activation='softmax'))

# # Compile the model
# The default number of epochs is 1, however we can experiment with different number of epochs, different loss functions, optimizers etc. 


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(X_train,y_train, validation_split=0.3)

# # Evaluating the Model
# Evaluate the trained model with X_test and y_test using one line of code


model.evaluate(X_test, y_test)

