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


def forward_prop(xtrain, weights, offsets, activation_fn, output_fn):
    '''
    Performs forward propagation on the training data with
    a matrix a weights, where weights[l] and offsets[l] 
    correspond to the weight and offset used to aggregate 
    neurons from layer l-1 to layer l.
    activation_fn is used on the hidden layers, and output_fn
    is used to compute the final output
    '''
    debug = 0   # set to true for more verbose output
    if debug:
        print 'xtrain', xtrain.shape
        print 'weights', len(weights), weights[0].shape
        print 'offsets', len(offsets), offsets[0].shape
    
    #count the number of layers in the network:
    L = len(weights) + 1
    #ensure that the number of offsets equals the number of weights
    assert len(weights) == len(offsets)
    
    # keep track of aggregated and activated vectors at each layer
    aggregated = []
    activated = []
    
    #compute neurons for the hidden layers
    prev_layer = xtrain
    for l in range(L-1):
        
        #aggregate the inputs from the previous layer
        W = weights[l]
        layer_l = np.dot(W.T, prev_layer) + offsets[l]
        if debug:
            print 'l: ', l ,'\t', 'wts', W.shape, '\t', 'b', offsets[l].shape
            print 'aggregated', layer_l, layer_l.shape
        aggregated.append(layer_l)
            
        #activate the neurons in the current layer
        if l == L-2:
            activate = output_fn
        else:
            activate = activation_fn
        layer_l = activate(layer_l)
        activated.append(layer_l)
        if debug:
            print 'activated', layer_l
        prev_layer = layer_l

    return aggregated, activated
                

    
def back_prop(weights, offsets, aggregated, activated):
    '''
    Computes the derivate of less with respect to aggregated value for each 
    layer of the neural nets using the recursive update rule 
    d[l] = Diag(f'(z[l]))W[l+1]
    with a base case given by 
    
    '''
    return None 
    
def ReLU(layer ,derivative = False):
    '''
    Given a vector of aggregated neuron values, applies the 
    ReLU activation function to return a vector of the same dimension
    where ReLU(z)=max(0,z)
    '''
    relu = 0.5*(layer + np.absolute(layer))
    if not derivative:
        return relu
    else:
        # replace all non-zero values with 1 
        relu[relu>0] = 1
        return relu
    
def softmax(layer, derivative = False):
    '''
    Given a k x 1 vector, computes the softmax probability
    output as a k x 1 vector
    '''
    e_x = np.exp(layer - np.max(layer))
    softmax_layer = e_x / e_x.sum(axis=0)
    # replace zeros with small values to avoid underflow
    softmax_layer[softmax_layer == 0] = 1e-10
    
    if not derivative:
        return softmax_layer
    else:
        return softmax_layer * (1 - softmax_layer)
    
def cross_entropy(expected, actual, derivative = False):
    '''
    Given an expected softmax'd output layer, computes the cross-entropy loss
    with the actual one-hot output layer, 
    where Loss = - sum(actual[i]*log[expected[i]])
     
    If specified, returns the derivative in column vector format
    '''
    debug = False
    
    # Ensure dimensions are correct    
    actual = actual.reshape(-1,1)
    expected = expected.reshape(-1,1)    
    assert actual.shape[0] == expected.shape[0]
    
    # Replace zeros in the expected vector to avoid underflow
    expected[expected == 0] = 1e-10
    
    # Compute log of expected values   
    log_expected = np.log(expected)
    if debug:
        print 'Expected', log_expected.shape, '\t', 'Actual', actual.shape 
    
    if not derivative:
        return -np.dot(actual.T,log_expected)
    else:
        inverse_expected = 1./expected
        return -np.multiply(inverse_expected.T, actual.T).T
    
    
def NN_train(Xtrain, Ytrain, L=3, M = None, k=3, activation_fn=ReLU, output_fn=softmax):
    '''
    Trains a neural network given training data Xtrain and Y train
    using the following parameters:
    L - number of layers (including input (l=1) and output (l=L) layers)
    M - array of number of neurons at each level. If unspecified, assume
        we will use the number of dimensions of the data.
    k - the number of output classes
    activation_fn - the activation function used. We will use the same 
                    activation function, ReLU for each neuron in the 
                    hidden layers
    '''
    n,d = Xtrain.shape # n is the number of data points, d is the dimension
        
    
    #Initialise the weights and offsets for each layer depending on the number 
    #of neurons per layer specified by M
    if M is None:
        M = [d]*(L-2)+[k]
        
    # Ensure we are specifying the correct number of weights
    assert len(M) == L-1
    
    weights = []
    # Create a weight matrix of dimensions m1 x m2, where m1 is the number of
    # neurons in layer l and m2 is the number of neurons in layer l+1
    
    offsets = []
    # Create an offset matrix of dimensions m1 * 1, where m1 is the number of 
    # neurons in layer l
    
    m1 = d
    for i in range(L-1):
        m2 = M[i]
        # Weight values are randomly initialized with mean 0 and std. dev 1/sqrt(m1)
        W = np.random.normal(0, 1./np.sqrt(m1), (m1,m2))
        weights.append(W)
        b = np.random.normal(0,1./np.sqrt(m1), (m1,1))
        offsets.append(b)
        m1 = m2
    
    # Train the neural network until its performance on a validation set plateaus
    converged = False    
    while not converged:
    # Propagate weights forward through neural network
    
    
    # 
    
    return None 
    
def predictNN(x, weights, offsets):
    '''
    '''
    
    return None

def classify_error(X, Y, weights, offsets):
    '''
    '''
    
    return None