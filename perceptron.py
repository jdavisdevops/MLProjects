import numpy as np
#import matplotlib
import sys
#import matplotlib.pyplot as plt
import time
#from helper import *

def perceptron_update(x, y, w):
    """
    function w=perceptron_update(x,y,w);

    Implementation of Perceptron weights updating
    Input:
    x : input vector of d dimensions (d)
    y : corresponding label (-1 or +1)
    w : weight vector of d dimensions

    Output:
    w : weight vector after updating (d)
    """

    # YOUR CODE HERE
    w = w + np.dot(y, x)
    return w
    # YOUR CODE HERE
    # raise NotImplementedError()


# little test
x = np.random.rand(10)
y = -1
w = np.random.rand(10)
w1 = perceptron_update(x, y, w)
N = 100;
d = 10;
xs = np.random.rand(N, d)
w0 = np.random.rand(2)
b0 = np.random.rand()*2-1
w = np.random.rand(1, d)
ys = np.sign(xs.dot(w0)+b0)
x = np.random.rand(N, d)
y = np.sign(w.dot(x.T))[0]


def perceptron(xs, ys):
    """
    function w=perceptron(xs,ys);

    Implementation of a Perceptron classifier
    Input:
    xs : n input vectors of d dimensions (nxd)
    ys : n labels (-1 or +1)

    Output:
    w : weight vector (1xd)
    b : bias term
    """

    n, d = xs.shape  # so we have n input vectors, of d dimensions each
    w = np.zeros(d)
    b = 0.0

    # YOUR CODE HERE
    count = 0
    while count <= 100:
        m = 0
        for i in np.random.permutation(n):
            print(ys[i])
            '''if ys[i]*(np.dot(w, xs[i]) + b) <= 0:
                perceptron_update(xs[i], ys[i], w)
                b += ys[i]
                m += 1'''

        if m == 0:
            break
        count += 1

    # YOUR CODE HERE
    # raise NotImplementedError()
    return (w, b)


print(perceptron(xs, ys))


def test_Perceptron1():
    N = 100;
    d = 10;
    x = np.random.rand(N,d)
    w = np.random.rand(1,d)
    y = np.sign(w.dot(x.T))[0]
    w, b = perceptron(x,y)
    preds = classify_linear_grader(x,w,b)
    return np.array_equal(preds.reshape(-1,),y.reshape(-1,))
def test_Perceptron2():
    x = np.array([ [-0.70072, -1.15826],  [-2.23769, -1.42917],  [-1.28357, -3.52909],  [-3.27927, -1.47949],  [-1.98508, -0.65195],  [-1.40251, -1.27096],  [-3.35145,-0.50274],  [-1.37491,-3.74950],  [-3.44509,-2.82399],  [-0.99489,-1.90591],   [0.63155,1.83584],   [2.41051,1.13768],  [-0.19401,0.62158],   [2.08617,4.41117],   [2.20720,1.24066],   [0.32384,3.39487],   [1.44111,1.48273],   [0.59591,0.87830],   [2.96363,3.00412],   [1.70080,1.80916]])
    y = np.array([1]*10 + [-1]*10)
    w, b =perceptron(x,y)
    preds = classify_linear_grader(x,w,b)
    return np.array_equal(preds.reshape(-1,),y.reshape(-1,))

test_Perceptron1()
