'''
----PROBLEM 1 ----
Use Python to implement a program to train fully-connected neural nets using stochastic
gradient descent. Do your implementation in sufficient generality so that you can vary the number
of nodes in each hidden layer and the number of hidden layers.

Pseudocode given in Lecture Notes:
1. Input - Training pair (x,y); set activations a_1 for layer 1
2. Feedforward: for each layer, l = 2,3,...L, compute:
                z_l = W_l.T*a_{l-1} + b_l
                a_l = f(z_l)
3. Output "error"
4. Backprop: for each layer, l = L-1,...,2 compute:
                d
'''

import numpy as np
import matplotlib
import pylab as pl


def forward_prop(Xtrain, weights, offsets, activation_fn, output_fn):
    '''
    Performs forward propagation on the training data with
    a matrix a weights, where weights[l] and offsets[l] 
    correspond to the weight and offset used to aggregate 
    neurons from layer l-1 to layer l.
    activation_fn is used on the hidden layers, and output_fn
    is used to compute the final output
    '''
    debug = False   # set to true for more verbose output
    if debug:
        print 'Xtrain', Xtrain.shape
        print 'weights', weights.shape, weights[0].shape
        print 'offsets', offsets.shape
    
    #count the number of layers in the network:
    L = len(weights) + 1
    #ensure that the number of offsets equals the number of weights
    assert len(weights) == len(offsets)
    
    #compute neurons for the hidden layers
    prev_layer = Xtrain
    for l in range(L-1):
        #aggregate the inputs from the previous layer
        W = weights[l]
        layer_l = np.dot(W.T, prev_layer) + b[l]
        if debug:
            print 'l: ', l
            print 'wts', W.shape
            print 'b', b[l]
            print 'aggregated', layer_l
        #activate the neurons in the current layer
        layer_l = activation_fn(layer_l)
        if debug:
            print 'activated', layer_l
        prev_layer = layer_l
        
    #compute the final output which has a different activation fn
    W = weights[L-1] 
    output_layer = np.dot(W.T, prev_layer) + b[L-1]
    #activate the neurons in the final layer
    output_layer = output_fn(output_layer)
    
    return output_layer
                

    
def back_prop():
    '''
    
    '''
    
def ReLU( ,derivative = False):
    '''
    
    '''

def softmax(layer):
    '''
    Given a k x 1 vector, computes the softmax probability
    output as a k x 1 vector
    '''
    e_x = np.exp(layer - np.max(layer))
    softmax_layer = e_x / e_x.sum(axis=0)
    
    return softmax_layer
    
def cross_entropy( ,derivative = False):
    '''
    '''
    
    
    
def NN_train(Xtrain, Ytrain, L=3, M = None, k=3, ):
    '''
    Trains a neural network given training data Xtrain and Y train
    using the following parameters:
    L - number of layers (including input (l=1) and output (l=L) layers)
    M - array of number of neurons at each level. If unspecified, assume
        we will use the number of training points.
    k - the number of output classes
    activation_fn - the activation function used. We will use the same 
                    activation function, ReLU for each neuron in the 
                    hidden layers
    '''
    