""" NN-L715 Spring 2023 Wk3 Practical Andrew Davis - Perceptron (https://cl.indiana.edu/~ftyers/courses/2023/Spring/L-715/practicals/perceptron/perceptron.html) """
""" Links Used: 
- https://pyimagesearch.com/2021/05/06/implementing-the-perceptron-neural-network-with-python/ 
- https://stats.stackexchange.com/questions/71335/decision-boundary-plot-for-a-perceptron """

#    First you randomly initialise the weights, W
#    For n epochs:
#        For each training example x, y in the training data X, Y
#            Multiply the feature vector x by the weight vector W and pass through the step function to get ŷ
#            If ŷ is not equal to y:
#                If ŷ is 0:
#                    add α · x to W
#                    update θ = θ + α
#                If ŷ is 1:
#                    subtract α · x to W
#                    update θ = θ - α

#Step 1: Import the necessary packages

import numpy as np
import matplotlib.pyplot as plt

#Step 2: Read in the file

IN = open('perceptron_training_data.tsv')
#OUT = open('perceptron_results_matplotlib.tsv')

text = IN.readlines()
X = []
first = True
y = []

for line in text:
    
    if first:
        first = False
        continue

    row = line.strip().split('\t')
    X.append([float(row[0]), float(row[1])])
    y.append(int(row[2]))
    
print(X)
print(y)

#Step 3: Create Perceptron Class

class Perceptron:
	#N is the number of columns in our input feature vectors ; alpha is the learning rate of the perceptron
    def __init__(self, N, learnrate=0.1): 
	    # initialize the weight matrix and store the learning rate
        # Line below fills our weight matrix W with random values sampled from a “normal” (Gaussian) distribution with zero mean 
        # and unit variance. The weight matrix will have N + 1 entries, one for each of the N inputs in the feature vector, 
        # plus one for the bias. We divide W by the square-root of the number of inputs, a common technique used to scale our 
        # weight matrix, leading to faster convergence.
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.learnrate = learnrate

    #Step 3b: Add Step that mimics the step equation, says if x is positive we return 1, if not return 0
	
    def step(self, x):
		# apply the step function
        return 1 if x > 0 else 0

    #Step 3b: Build a fit to train the model

    def fit(self, X, y, epochs=25):
        # The X value is our actual training data. The y variable is our target output class labels (i.e. what our network should be 
        # predicting). Finally, we supply epochs, the number of epochs our Perceptron will train for.
		# insert a column of 1's as the last entry in the feature
		# matrix -- this little trick allows us to treat the bias
		# as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        #Step 3c: For loop describing actual training procedure

		# loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
			# loop over each individual data point
            for (x, target) in zip(X, y):
				# take the dot product between the input features
				# and the weight matrix, then pass this value
				# through the step function to obtain the prediction ; p is like a prediction
                p = self.step(np.dot(x, self.W))
				# only perform a weight update if our prediction
				# does not match the target
                if p != target:
					# determine the error
                    error = p - target
					# update the weight matrix
                    self.W += -self.learnrate * error * x

#Step 3d: Build a predict function that predicts class labels
    def predict(self, X, addBias=True):
		# ensure our input is a matrix
        X = np.atleast_2d(X)
		# check to see if the bias column should be added
        if addBias:
			# insert a column of 1's as the last entry in the feature
			# matrix (bias)
            X = np.c_[X, np.ones((X.shape[0]))]
		# take the dot product between the input features and the
		# weight matrix, then pass the value through the step
		# function
        return self.step(np.dot(X, self.W))

#Step 4: TIME TO IMPLEMENT

#Step 4b: construct the OR dataset
X = np.array(X)
y = np.array(y)

#Step 4c: define our perceptron and train it -- ALPHA IS OUR LEARNING RATE (I changed alpha to learnrate)

print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], learnrate=0.1)
p.fit(X, y, epochs=25)

#Step 4d: Now that our perceptron is trained we can evaluate it
print("[INFO] testing perceptron...")

#Step 4e: Now that our network is trained, loop over the data points
for (x, target) in zip(X, y):

	# make a prediction on the data point and display the result
	# to our console
	pred = p.predict(x)
	print("[INFO] input_data={}, true_label={}, perceptron_prediction={}".format(x, target, pred))

#for (x, target, pred) in

accuracy = correct/len(df) * 100 #Need to define a correct
print('=========EPOCH========= {}'.format(epoch + 1))
print('Accuracy: {}%'.format(accuracy))
print('Weights:', W)
print('Bias:', bias)
print('Decision Boundary:', decision_boundaries[-1])
print() 

IN.close()
#OUT.close()

#Step 5 VISUALIZE THE DATA

def plot_data(self, x, y, weights):
    #Figure configuration
    plt.figure(figsize=(10,6))
    plt.grid(True)

    #plot input samples(2D data points) and i have two classes. 
    #one is +1 and second one is -1, so it red color for +1 and blue color for -1
    for input,target in zip(x,targets):
        plt.plot(input[0],input[1],'ro' if (target == 1.0) else 'bo')

    # Here i am calculating slope and intercept with given three weights
    for i in np.linspace(np.amin(x[:,:1]),np.amax(x[:,:1])):
        slope = -(weights[0]/weights[2])/(weights[0]/weights[1])  
        intercept = -weights[0]/weights[2]

        #y =mx+c, m is slope and c is intercept
        y = (slope*i) + intercept
        plt.plot(i, y,'ko')

plt.show()