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
import pylab as pl

    
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

def forward_prop(xtrain, weights, offsets, activation_fn = ReLU, output_fn = softmax):
    '''
    Performs forward propagation on the training data with
    a matrix a weights, where weights[l] and offsets[l] 
    correspond to the weight and offset used to aggregate 
    neurons from layer l-1 to layer l.
    activation_fn is used on the hidden layers, and output_fn
    is used to compute the final output
    '''
    debug = 0 # set to true for more verbose output
    if debug:
        print 'xtrain', xtrain.shape
        print 'weights', len(weights), weights[0].shape
        print 'offsets', len(offsets), offsets[0].shape
    
    #count the number of layers in the network:
    L = len(weights) + 1
    #ensure that the number of offsets equals the number of weights
    assert len(weights) == len(offsets)
    
    #compute neurons for the hidden layers
    prev_layer = xtrain.reshape(-1,1)
        
    # keep track of aggregated and activated vectors at each layer
    aggregated = []
    activated = []
    
    for l in range(L-1):
        #aggregate the inputs from the previous layer
        W = weights[l]
        if debug:
            print 'l: ', l ,'\t', 'wts', W.shape, '\t', 'b', offsets[l].shape, '\t', 'z', prev_layer.shape
        layer_l = np.dot(W.T, prev_layer) + offsets[l]
        if debug:
            print 'aggregated', layer_l
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
                

    
def back_prop(ytrain, weights, offsets, aggregated, activated, output_fn = softmax, activation_fn = ReLU, loss_fn = cross_entropy):
    '''
    Computes the derivate of less with respect to aggregated value for each 
    layer of the neural nets using the recursive update rule 
    d[l] = Diag(f'(z[l]))W[l+1]
    with a base case given by d[L] = Diag(f'(z[L])) * dLoss/da[L]
    
    Returns a list of error vectors , d   
    '''
    debug = 0
    #count the number of layers in the network:
    L = len(weights) + 1
    
    #initialise list of error vectors
    d = [0]*(L-1)
    
    base_case = True
    for l in range(L-2,-1,-1):
        #base case is the final layer of the network
        if base_case:
            #compute derivative of output function wrt aggregated layer
            z_l = aggregated[l]
            f_prime = output_fn(z_l, derivative = True).reshape(1,-1)
            diag_f = np.diagflat(f_prime)
            if debug:
                print 'basecase', f_prime, diag_f
            #compute derivative of loss wrt activated layer
            a_l = activated[l]
            l_prime = loss_fn(ytrain, a_l, derivative = True)
            d[l] = np.dot(diag_f, l_prime)
            base_case = False
        # all other errors can be computed in terms of subsequent errors
        else:
            W = weights[l+1]
            err = d[l+1]
            z_l = aggregated[l]
            # compute derivative wrt aggregated layer
            f_prime = activation_fn(z_l, derivative = True)
            diag_f = np.diagflat(f_prime)
            d_l = np.dot(np.dot(diag_f,W),err)
            d[l] = d_l            

    return d
   
    
def NN_train(Xtrain, Ytrain, Xval, Yval, L=3, M = None, k=3, learning_rate = 1e10, activation_fn=ReLU, output_fn=softmax, loss_fn = cross_entropy):
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
    
    #Ensure output layer has k neurons
    M[-1] = k
        
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
        b = np.random.normal(0, 1./np.sqrt(m1), (m2,1))
        offsets.append(b)
        m1 = m2
    print 'Initial Accuracy', classify_accuracy(Xval, Yval, weights, offsets), classify_accuracy(Xtrain, Ytrain, weights, offsets)
    # Train the neural network until its performance on a validation set plateaus
    converged = False 
    prev_acc = 0
    num_iters = 0
    while not converged:
        num_iters += 1
        # Choose random index for stochastic gradient update
        index = np.random.randint(0,n)
        xtrain = Xtrain[index].reshape(-1,1)
        ytrain = Ytrain[index]
        # Propagate weights forward through neural network
        aggregated, activated = forward_prop(xtrain, weights, offsets)
        # Compute error vectors through back propagation
        delta = back_prop(ytrain, weights, offsets, aggregated, activated)
        activated.append(xtrain)        
        # Perform gradient update for each set of parameters
        for l in range(L-1):
            weights[l] = weights[l] - learning_rate*np.dot(activated[l-1],delta[l].T)
            offsets[l] = offsets[l] - learning_rate*delta[l]
        # Test for convergence
        acc = classify_accuracy(Xval, Yval, weights, offsets)
        if acc<prev_acc-1e-2:
            converged = True
        prev_acc = acc
        test_acc = classify_accuracy(Xtrain, Ytrain, weights, offsets)
        if num_iters%100==0:
            print 'Iters, acc', num_iters, acc, test_acc

    return weights, offsets , acc, num_iters
    
def NN_predict(x, weights, offsets):
    '''
    Given the weights and offsets calculated from training the neural net,
    predict the class of x
    '''
    aggregated, activated = forward_prop(x, weights, offsets)
    y = activated[-1]
    # convert predicted vector to one-hot classification
    y[y.argmax()] = 1
    y[y < 1] = 0
    # what if there are more than one max values?
    if sum(y) > 1:
        y[y.argmax()[0]] = 1
    return y

def classify_accuracy(X, Y, weights, offsets):
    '''
    Given a set of X and Y values, as well as weights and offsets calculated by
    training the neural net, compute the error rate
    '''
    #Check dimensions for sanity
    n1, d = X.shape
    n2, k = Y.shape
    assert n1==n2
    correct = 0.0
    for i in range(len(X)):
        predict_y = NN_predict(X[i], weights, offsets).reshape(1,-1)
        y = Y[i].reshape(1,-1)
        correct += np.dot(y, predict_y.T)[0][0]
    return correct/n1

def one_hot(Y,k):
    '''
    Given an array of target values, Y, and number of labels, k
    convert the array to an array of one-hot vectors
    '''
    n = Y.shape[0] #number of data points
    one_hot = np.zeros((n, k))
    one_hot[np.arange(n), Y] = 1
    return one_hot
    
#### TEST ON TOY DATASET ####
test_toy = True
if test_toy:
    toy_data = './hw3_resources/data/data_3class.csv'
    train = np.loadtxt(toy_data)
    X = train[:,0:2]
    Y = train[:,2:3].astype(int)
    Xtrain = X[:400,:]
    Ytrain = one_hot(Y[:400,:].reshape(1,-1)[0],3)

    Xval = X[400:600,:]
    Yval = one_hot(Y[400:600,:].reshape(1,-1)[0],3)
    print 'Training...'
    weights, offsets , acc, num_iters = NN_train(Xtrain, Ytrain, Xval, Yval, L=4,M=[10,10,3])
    print 'Finished training in ', num_iters, ' rounds with a validation accuracy of ', acc