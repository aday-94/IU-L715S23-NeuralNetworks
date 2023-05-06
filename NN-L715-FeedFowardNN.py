""" NN-L715 Spring 2023 Wk7 & 8 Practical Andrew Davis - FeedForwardNN """

""" Links Used: 
- Killer Combo: Softmax and Cross Entropy (https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba)
- Understanding and implementing Neural Network with SoftMax in Python from scratch (https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/)
- Back-Propagation simplified (https://towardsdatascience.com/back-propagation-simplified-218430e21ad0)
- A worked example of backpropagation (https://alexander-schiendorfer.github.io/2020/02/24/a-worked-example-of-backprop.html)
- Sigmoid, Softmax and their derivatives (https://themaverickmeerkat.com/2019-10-23-Softmax/)
- Classification and Loss Evaluation - Softmax and Cross Entropy Loss (https://deepnotes.io/softmax-crossentropy) """

import re 
import numpy as np

np.random.seed(3)

test_text = '''where are you?
is she in mexico?
i am in greece.
she is in mexico.
is she in england?
'''

train_text = '''are you still here?
where are you?
he is in mexico.
are you tired?
i am tired.
are you in england?
were you in mexico?
is he in greece?
were you in england?
are you in mexico?
i am in mexico.
are you still in mexico? 
are you in greece again?
she is in england.
he is tired.
'''

""" Step 1: Define Functions -- Tokenise text, Create one-hot vectors, Create ReLU Activation and Softmax Activation Functions
X is the input data, Z is the  """

def tokenise(s):
    return [i for i in re.sub('([.?])', ' \g<1>', s).strip().split(' ') if i]

def one_hot(y, classes):
    onehot = np.zeros((len(y), classes)) # creates matrix of ? rows, ? columns 
    
    # Iterate through y and update onehot's column to 1 based on the class
    # y [0, 1, 4, 3, 2]
    for i, v in enumerate(y):
        onehot[i][v] = 1
    return onehot

"""
X is the input data
Z is the data after the weights and bias have been added
"""

def relu(X): # You're literally just comparing it with 0 and if its a negative number 0 is bigger otherwise its smaller
    return 0 if X < 0 else X
    #return np.maximum(0, X) # This was Matt's option
    # FYI we may want to drop relu and use sigmoid instead

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def softmax(Z):
    return (np.exp(Z)/np.exp(Z).sum()) #(can do the above in one line)

def predict(X, w1, w2, bias):
    #First Layer (Hidden layer ; (X*w1 +B))
    pre1 = np.dot(X, w1) + bias
    #Activation function on first (hidden) layer
    activation1 = sigmoid(pre1)
    #Second Layer (Output Later ; activation1 * w2)
    pre2 = np.dot(activation1, w2)
    #Activation function on second (output) layer
    y_hat = softmax(pre2)

    #Returning output layer and hidden layer 
    return y_hat, activation1

def generate(w1, w2, bias, prefix):
    ids = []
    for token in prefix:
        ids.append(word2idx[token])
    X = np.array(ids)
    y_hat, act1 = predict(X, w1, w2, bias)

    return idx2word[np.argmax(y_hat)]

"""
P is the output layer
Y is the target labels
"""

def cross_entropy(prediction, true_label):
    losses = np.multiply(true_label, np.log(prediction))
    total_loss = -np.sum(losses/true_label.shape[0])
    return losses, total_loss
    """ m = Y.shape[0]
#    print("------Cross Entropy------")
#    print(f"---P{P[:1]}")
#    print(f"---Y{Y[:1]}")
#    print(f"---P[0:m, Y]{P[0:m, Y]}")
    log_likelihood = -np.log(P[range(m), Y]) # calculate log likelihood by taking the negative log of the y_i column of P
#    print(f"---Log Likelihood{log_likelihood[:1]}")
    loss = np.sum(log_likelihood) / m # calculate loss by summing values in log_likelihood and deviding by number of examples
#    print("-------------------------")
    return loss """

""" Step 2: Process the text, Tokenize the text and add the Tokens to a list called vocab.
Then, create two dictionaries one with the word as the key and the id as the value (word2idx) and then another dictionary 
that's vice versa (idx2word) """

vocab = list(set([token for token in re.sub('([.?])', ' \g<1>', train_text)
             .replace(' ', '\n').strip().split('\n') if token]))
vocab += ['<BOS>', '<EOS>', '<PAD>']
vocab.sort()

#print(vocab)

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

print(word2idx)
print(idx2word)

""" Step 3: Now, pad the data so the matrices/vectors are all the same length(?), create a list called train sentences,
create a for loop that takes the the train text, processes it into sentences, tokenizes the sentences, pads them 
so they'll be the same length, then we append the processed sentences from the train text to the train sentences list. """

pad = max([len(tokenise(i)) for i in train_text.split('\n')]) + 1
print(pad)
train_sentences = []
for line in train_text.strip().split('\n'):
        tokens = tokenise(line)
        padded = ['<BOS>'] + tokens + ['<EOS>'] + ['<PAD>'] * (pad - len(tokens))
        train_sentences.append([word2idx[token] for token in padded])

""" X = []
y = []

for sentence in train_sentences:
    for i in range(pad - 2):
            X.append([sentence[i], sentence[i+1]])
            y.append(sentence[i+2])
                  
X = np.array(X)
X = np.append(X, np.ones((X.shape[0], 1)), axis = 1)
Y = np.array(y)
Yo = one_hot(Y, len(vocab))
X = np.vstack([X, [1,1]])

print(X.shape)
print(Y.shape)
print(Yo.shape)

ihw = np.random.randn(2,7) # These are the weights from x (input) to the hidden layer
how = np.random.randn(7, len(vocab)) # These are the weights from the hidden layer to output

E = np.random.randn(2, 4)
h = np.random.randn(4, 6) 
O = np.random.randn(6, len(vocab)) 
print(E.shape)
print(h.shape)
print(O.shape) """

""" Step 4: Create your list of training sentences and of testing sentences """

X = []
y = []

for sentence in train_sentences:
    for i in range(pad - 2):
            X.append([sentence[i], sentence[i+1]])
            y.append(sentence[i+2])
                  
X = np.array(X)
X = np.append(X, np.ones((X.shape[0], 1)), axis = 1) #bias now incorporated to X
Y = np.array(y)
Yo = one_hot(Y, len(vocab))

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Yo shape:", Yo.shape)
w1 = np.random.randn(2, 6)
w2 = np.random.randn(6, len(vocab))


#E = np.random.randn(2, 4)
h = np.random.randn(3, 6) 
o = np.random.randn(6, len(vocab)) 

#print(E.shape)
print("h shape:", h.shape)
print("o shape:", o.shape)

hidden_layer = sigmoid(X @ h)
output_layer = softmax(hidden_layer @ o)
print("Output Layer Shape:", output_layer.shape)
cross_entropy_loss = cross_entropy(output_layer, Y)
alpha_learn = 0.01
epochs = 25


test1 = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
test2 = list([1, 2, 1, 2, 1])
#print(test1[0:5, test2])

""" Step 5: Create your variables for your model (learn rate, bias, etc) & then create your for loop """

#Jeremy's training loop

epochs = 25
all_losses = []

for i in range(epochs):
    y_hat, activation1 = predict(X, w1, w2, bias)
    losses, total_loss = cross_entropy(y_hat, Yo)
    all_losses.append(total_loss)

    diff = y_hat - Yo
    der_L_w2 = np.transpose(np.transpose(diff) @ activation1)
    der_l_w1 = np.transpose(diff @ w2.T) @ X
    der_bias = np.sum(np.transpose(diff @ w2.T) axis = 1)

    w1 -=
    w2 -=
    bias -=

next_word = generate(w1, w2, bias, ['am', 'in'])

