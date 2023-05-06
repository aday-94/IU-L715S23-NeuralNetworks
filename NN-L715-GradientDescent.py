""" NN-L715 Spring 2023 Wk4 & 5 Practical Andrew Davis - Gradient Descent (https://cl.indiana.edu/~ftyers/courses/2023/Spring/L-715/practicals/gradient/gradient.html) """

""" Gradient descent

The last time we looked at the perceptron, which has a simple learning algorithm, where we adjust the weights by the 
difference between the expected output and the predicted output. In this exercise we will look at a more sophisticated 
learning algorithm called gradient descent. We will test it with simple linear regression. """

""" Generating some data

First generate some training data in the form of (X, Y) pairs. You can use np.random.rand and np.random.randn for this. 
Set your random seed in numpy to 3 before generating the data so that the results will be comparable.

Now plot the data using matplotlib. """

import numpy as np
import matplotlib.pyplot as plt
import random
np.random.seed(3)

def generate_data(n):
    X = 2 * np.random.rand(n,1)
    Y = 4 + 3 * X + np.random.rand(100,1)
    return X, Y

n = 100
X, Y = generate_data(n)

#Now, we plot the data

plt.scatter(X, Y)
plt.show()

""" Algorithm

For the algorithm we need two things:

A cost function, J. This tells you how far you are from the desired result. This is like the difference between the expected output 
and the actual output in the perceptron algorithm.

We calculate the cost by creating a function which takes three arguments:

    The weights, W,
    the inputs X and
    the expected outputs y

First create the predicted output ŷ by taking the dot product of the input and the weights. Then calculate the cost using 
the following formula:

    cost = (1/2m) * Σ (ŷ - y)²

Where m is the number of training examples.

The gradient descent algorithm is then:

    for each epoch
        calculate the prediction, ŷ
        update the weights using the gradient of the cost function

To update the weights you subtract the multiplication of the learning rate by the partial derivative (dJ/dW) of the weights.

    W = W - (α * (dJ/dW))

The learning rate you should be familiar with from the perceptron exercise. """

""" Exercise 1:

    Implement the gradient descent algorithm
    Plot the cost over epochs for three different learning rates, 0.1, 0.01, 0.001 """

#Step 1: Create a cost function

def cost_function(W, X, y, y_hat, m):
    
    error = np.sum(np.square(y_hat - y))
    cost = (1/(2*m)) * error
    
    return cost

#Step 2: Create your Gradient Descent Function (X is dataset, y is target prediction/label, epochs for training and alpha is the learnrate)

def gradient_descent(X, y, epochs, alpha_learn):
    
    W = np.random.rand()
    cost_per_epoch = []

    #Step 2A: Our Training Loop that runs the assigned amount of epochs
    for i in range(epochs):
        
        #Step 2B: Calculating our prediction/predicted output (that's what y_hat is, W is weights)
        y_hat = np.dot(X, W)
        
        #Step 2C: m is the shape of the dataset (dataset is X, m is matrix)
        m  = X.shape[0]
        
        #Step 2D: Now, we calculate the cost with the function we built earlier (J is our cost per epoch of training)
        J = cost_function(W, X, y, y_hat, m)
        cost_per_epoch.append(J)
        
        #Step 2E: Now, we update weights our weights
        derivative = np.sum((1/m) * X *(y_hat - y))
        W = W - (alpha_learn * derivative)  
    
    return W, cost_per_epoch

#Step 3: Now, it's time to use the algorithm, first we'll chose the number of epochs, our learning rate (alpha), etc

epochs = 150

#Step 3A: Test 1, learning rate of 0.1

alpha_learn1 = 0.1
W1, cost_per_epoch1 = gradient_descent(X, Y, epochs, alpha_learn1)

plt.plot(cost_per_epoch1)
plt.show()

#Step 3B: Test 2, learning rate of 0.01

alpha_learn2 = 0.01
W2, cost_per_epoch2 = gradient_descent(X, Y, epochs, alpha_learn2)

plt.plot(cost_per_epoch2)
plt.show()

#Step 3C: Test 3, learning rate of 0.001

alpha_learn3 = 0.001
W3, cost_per_epoch3 = gradient_descent(X, Y, epochs, alpha_learn3)

plt.plot(cost_per_epoch3)
plt.show()

""" Extensions

Stochastic gradient descent

In normal gradient descent we process the examples in the training set in order. This can lead to local minima if for example 
the examples are in some particular order. An alternative is to pick the examples randomly (hence the stochastic in the name).

One way of doing this is to randomly sort the training examples before each epoch and then update the weights after each training 
example. """

""" Exercise 2:

    Update your implementation to use stochastic gradient descent """

#Step 1: Let's change our gradient descent function to go through training examples randomly instead of in order (hence stochastic)

def stochastic_gradient_descent(X, y, epochs, alpha):
    #Step 1A: First, we randomly shuffle the input data by:
    #Finding the shape of the dataset (m is for matrix, s is for shape)
    m  = X.shape[0]
    s = np.arange(m)
    
    W = np.random.rand()
    cost_per_epoch = []

#     j = 0
    #Step 1B: Now, we create our for training loop to run the assigned number of epochs:
    for i in range(epochs):
        
        #So, we take one example at a time (j is our chosen, random example)
        j =  random.choice(s)
#         j = j%100
        
        #Then, we make a prediction / calculate a predicted output based on j (the random training example)
        y_hat = np.dot(X[j], W)
        
        #Then, calculate the cost (J is the cost for each epoch of training based on little j [how it's different than exercise 1])
        J = cost_function(W, X[j], y[j], y_hat, m)
        cost_per_epoch.append(J)
        
        #Finally, we have to update our weights before we do the next epoch of training
        derivative = np.sum((1/m) * X[j] *(y_hat - y[j]))
        W = W - (alpha * derivative)  
    
#         j+=1
    return W, cost_per_epoch

#Step 2: Now it's time to use the algorithm

#Step 2A: Test 1, learning rate of 0.1

W4, cost_per_epoch4 = stochastic_gradient_descent(X, Y, epochs, alpha_learn1)

plt.plot(cost_per_epoch4)
plt.show()

#Step 2B: Test 2, learning rate of 0.01

W5, cost_per_epoch5 = stochastic_gradient_descent(X, Y, epochs, alpha_learn2)

plt.plot(cost_per_epoch5)
plt.show()

#Step 2C: Test 3, learning rate of 0.001

W6, cost_per_epoch6 = stochastic_gradient_descent(X, Y, epochs, alpha_learn3)

plt.plot(cost_per_epoch6)
plt.show()

""" Minibatch gradient descent

Another improvement to gradient descent is to use random samples, but instead of processing all of the examples at once, 
process a few examples at the same time. This means that we don't calculate the gradients for a all of the training examples, 
but rather calculate them for a batch of training examples. """

""" Exercise 3:

    Update your implementation to use minibatch gradient descent. Use a batch size of 5 to start out with. """

#Step 1: Let's change our gradient descent function to go through mini batches of training examples instead of all at once, we'll start with mini batch of 5

def minibatch_gradient_descent(X, y, epochs, alpha, batch):
    
    #Step 1A: First we find the shape of the data and take batches instead of the entire dataset
    #Finding the shape of the dataset (m is for matrix)
    m  = X.shape[0]
    
    if batch > m:
        print(' Batch size out of bound')
        return 0,[]

    W = np.random.rand()
    cost_per_epoch = []

    j = 0
    k = batch
    
    #Step 1B: Now, we create our for training loop to run the assigned number of epochs:

    for i in range(epochs):

        # Defining the batch sizes
        j = j%m
        k = k%m
        
        # Calculating predicted output
        y_hat = np.dot(X[j:k], W)
        
        # Calculating the cost
        J = cost_function(W, X[j:k], y[j:k], y_hat, m)
        cost_per_epoch.append(J)
        
        # Updating weights
        derivative = np.sum((1/m) * X[j:k] *(y_hat - y[j:k]))
        W = W - (alpha * derivative)
        
        j += batch
        k += batch
    
    return W, cost_per_epoch
