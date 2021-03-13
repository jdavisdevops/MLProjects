import numpy as np
import os

n = 20
d = 5
X = np.random.rand(50,15) # generate n random vectors with d dimensions
w = np.random.rand(15)
b = np.random.rand(1)
def sigmoid(z):
    # Input:
    # z : scalar or array of dimension n
    # Output:
    # sgmd: scalar or array of dimension n

    # YOUR CODE HERE
    sgmd = 1/(1+np.exp(-z))
    # YOUR CODE HERE
    #raise NotImplementedError()

    return sgmd

def y_pred1(X, w, b=0):
    # Input:
    # X: nxd matrix
    # w: d-dimensional vector
    # b: scalar (optional, if not passed on is treated as 0)
    # Output:
    # prob: n-dimensional vector

    # YOUR CODE HERE
    n,d = X.shape
    probs = []
    for i in range(n):
        probs.append(sigmoid((np.dot(w,X[i])+b)))
    prob = np.ones(n)
    prob = probs * prob
    # YOUR CODE HERE
    #raise NotImplementedError()

    return prob

def ypred(X,w,b=0):
    prob = sigmoid(X @ w + b)
    sh1 = X.shape
    sh2 = w.shape
    print(prob,sh1,sh2)
print(ypred(X,w,b))
#test = ypred(X,w,b)
#print(test.shape)
y = (np.random.rand(500)>0.5)*2-1; # define n random labels (+1 or -1)
def log_loss1(X, y, w, b=0):
    # Input:
    # X: nxd matrix
    # y: n-dimensional vector with labels (+1 or -1)
    # w: d-dimensional vector
    # Output:
    # a scalar
    assert np.sum(np.abs(y))==len(y) # check if all labels in y are either +1 or -1
    # YOUR CODE HERE
    pred = y_pred(X,w,b=0)
    ypred = y * pred
    #nll = -np.sum(np.log(pred))
    print(pred, y)
    # YOUR CODE HERE
    #raise NotImplementedError()

    #return nll
print(log_loss(X,y,w,b=0))
def log_loss(X, y, w, b=0):
    # Input:
    # X: nxd matrix
    # y: n-dimensional vector with labels (+1 or -1)
    # w: d-dimensional vector
    # Output:
    # a scalar
    assert np.sum(np.abs(y))==len(y) # check if all labels in y are either +1 or -1
    # YOUR CODE HERE
    #nll = -np.sum(np.log(y_pred(X,w,b))) original
    n,d = X.shape
    probs = []
    for i in range(n):
        probs.append(sigmoid(y[i]*(np.dot(w.T,X[i])+b)))
    prob = np.ones(n)
    prob = probs * prob

    nll = -np.sum(np.log(y_pred(X, w, b)))

    # YOUR CODE HERE
    #raise NotImplementedError()

    return nll

#print(log_loss(X,y,w,b))
X = np.random.rand(25,5) # generate n random vectors with d dimensions
w = np.random.rand(5) # define a random weight vector
b = np.random.rand(1) # define a bias
y = (np.random.rand(25)>0.5)*2-1 # set labels all-(+1)
def gradient(X, y, w, b):
    # Input:
    # X: nxd matrix
    # y: n-dimensional vector with labels (+1 or -1)
    # w: d-dimensional vector
    # b: a scalar bias term
    # Output:
    # wgrad: d-dimensional vector with gradient
    # bgrad: a scalar with gradient
    
    n, d = X.shape
    wgrad = np.zeros(d)
    bgrad = 0.0
    # YOUR CODE HERE
    wgrad = -y @ (sigmoid(-y*(np.dot(w,X.T)+b)))*X
    bgrad = np.sum(-y @ sigmoid(-y*(np.dot(w,X.T)+b)))
        
    # YOUR CODE HERE
    #raise NotImplementedError()
    return wgrad, bgrad
print(gradient(X,y,w,b))
