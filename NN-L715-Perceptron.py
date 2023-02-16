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

""" Step 1: Import the necessary packages """

import numpy as np
from matplotlib import pyplot as plt, lines, animation

""" Step 2: create list of train data, positive values, negative values, decision boundaries, generate your weights & bias, and choose
your learnrate (alpha) and the number of epochs you want to train """

train_data = [] #List perceptron training data
train_data_pos = [] #List of perceptron training data with 1 output values
train_data_neg = [] #List of perceptron training data with 0 output values
decision_boundaries = [] #List of decision boundary values
weights = np.random.rand() #Generate the weights
bias = np.random.rand(2) #Generate the bias
alpha_learnrate = 0.1 #Generate the learn rate (commonly termed as alpha)
epochs = 25 #Num of epochs you want to train

""" Step 3: Open your file, process the text (separated by tabs, and append it to train data """

with open('perceptron_training_data.tsv') as data_file:
    next(data_file)
    for line in data_file:
        l = line.strip('\n').split('\t')
        l[0] = float(l[0])
        l[1] = float(l[1])
        l[2] = int(l[2])
        train_data.append(l)

for line in train_data:
    if line[2]:
        train_data_pos.append(line)
    else:
        train_data_neg.append(line)

""" Step 4: Set the minimum & maximum value of x & y for your decision boundary function, then build you decision boundary function 
& get the decision boundary by calling the function """

minx = min([row[0] for row in train_data]) # the minimum value for feature 1
maxx = max([row[0] for row in train_data]) # the maximum value for feature 1
miny = min([row[1] for row in train_data]) # the minimum value for feature 2
maxy = max([row[1] for row in train_data]) # the maximum value for feature 2

def get_decision_boundary():
    global decision_boundaries
    y0 = (-bias - minx * weights[0]) / weights[1]
    y1 = (-bias - maxx * weights[1]) / weights[1]
    decision_boundaries.append([y0, y1])

get_decision_boundary()

""" Step 5: Write your For loop to train the model (adjust parameters such as epochs, bias, weights etc in Step 2) """

for epoch in range(epochs):

    correct = 0

    for row in train_data:

        x = [row[0], row[1]] #Select your data points ; y_hat is your prediction
        y_hat = 0.0

        if (np.dot(x, weights) + bias) > 0:
            y_hat = 1.0 #Set your prediction to 1

        if y_hat == row[2]: #If your prediction is correct, add a count to your correct
            correct += 1
        else: #If your prediction is incorrect, adjust the learnrate, bias etc for the next epoch for either a negative or positive, incorrect prediction
            if y_hat == 0:
                weights = np.add(weights, [alpha_learnrate * value for value in x])
                bias += alpha_learnrate
            elif y_hat == 1:
                weights = np.subtract(weights, [alpha_learnrate * value for value in x])
                bias -= alpha_learnrate

    get_decision_boundary() #Get the decision boundary points from each epoch

    accuracy = correct/len(train_data) * 100 #Calculate the accuracy based on the correct count
    print('=========EPOCH========= {}'.format(epoch + 1))
    print('Accuracy: {}%'.format(accuracy))
    print('Weights:', weights)
    print('Bias:', bias)
    print('Decision Boundary:', decision_boundaries[-1])
    print() #Print the Epoch, accuracy, weights, bias, and decision boundary for each epoch


    if correct == len(train_data): #If 100% correct, break the loop, no need to continue training for more epochs
        break

""" Step 6: Plot your results on matplotlib """

fig, ax = plt.subplots()
def acb(i):
    ax.clear()
    ax.set_xlim([minx - 0.5, maxx + 0.5])
    ax.set_ylim([miny - 0.5, maxy + 0.5])
    ax.set_title(f'Epoch{i+1}')
    ax.scatter([j[0] for j in train_data_pos], [k[1] for k in train_data_pos], c = 'red')
    ax.scatter([j[0] for j in train_data_neg], [k[1] for k in train_data_neg], c = 'blue')
    ax.plot([minx, maxx], decision_boundaries[i])
anim = animation.FuncAnimation(fig, acb, frames = len(decision_boundaries))
plt.show()

data_file.close()

""" End Program """