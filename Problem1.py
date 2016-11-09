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
import plotBoundary as plot

test = False

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
if test:
    layer = np.array([1,2,-1,0,4,5,1])
    print layer
    print ReLU(layer)
    print ReLU(layer, derivative = True)
    
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
        
if test:
    layer = np.array([1,2,-1,0,4,5,1])
    layer2 = layer + 100000
    print layer
    print softmax(layer)
    print softmax(layer2)
    print softmax(layer, derivative = True)
    
def cross_entropy(predicted, actual, derivative = False):
    '''
    Given an expected softmax'd output layer, computes the cross-entropy loss
    with the actual one-hot output layer, 
    where Loss = - sum(actual[i]*log[predicted[i]])
     
    If specified, returns the derivative in column vector format
    '''
    debug = False
    
    # Ensure dimensions are correct    
    actual = actual.reshape(-1,1)
    predicted = predicted.reshape(-1,1)  
    assert actual.shape[0] == predicted.shape[0]
    
    # Replace zeros in the expected vector to avoid underflow
    predicted[predicted == 0] = 1e-10
    
    # Compute log of expected values   
    log_predicted = np.log(predicted)
    if debug:
        print 'Predicted', log_predicted.shape, '\t', 'Actual', actual.shape 
    
    if not derivative:
        return -np.dot(actual.T,log_predicted)
    else:
        inverse_predicted = 1./predicted
        return -np.multiply(inverse_predicted.T, actual.T).T

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
        if np.isnan(layer_l).any():
            print W.T,'weights'
            print prev_layer, 'prev'
            print offsets[l], 'off'
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
#            #compute derivative of output function wrt aggregated layer
#            z_l = aggregated[l]
#            f_prime = output_fn(z_l, derivative = True).reshape(1,-1)
#            diag_f = np.diagflat(f_prime)
#            if debug:
#                print 'basecase', f_prime, diag_f
#            #compute derivative of loss wrt activated layer
            a_l = activated[l]
#            l_prime = loss_fn(a_l, ytrain, derivative = True)
#            d[l] = np.dot(diag_f, l_prime)
            
            #short cut to compute delta at output layer
            d[l] = a_l - ytrain.reshape(-1,1)
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
   
    
def NN_train(Xtrain, Ytrain, Xval, Yval, L=3, M = None, k=3, initial_rate = 0.005, fixed=False, activation_fn=ReLU, output_fn=softmax, loss_fn = cross_entropy):
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
    print 'Initial Accuracy', classify_accuracy(Xtrain, Ytrain, weights, offsets)
    best_weights = np.copy(weights)
    best_offsets = np.copy(offsets)
    # Train the neural network until its performance on a validation set plateaus

    history =5000 # number of previous accuracies to consider
    accuracies = [0]*history
    max_acc = 0
    num_iters = 0
    learning_rate = initial_rate
    while True:
        num_iters += 1
        if not fixed:
            learning_rate = initial_rate/np.power(num_iters, 1./3)
        # Choose random index for stochastic gradient update
        index = np.random.randint(0,n)
        xtrain = Xtrain[index].reshape(-1,1)
        ytrain = Ytrain[index]
        # Propagate weights forward through neural network
        aggregated, activated = forward_prop(xtrain, weights, offsets)
        # Compute error vectors through back propagation
        delta = back_prop(ytrain, weights, offsets, aggregated, activated)
        activated.append(xtrain)  
        if 0:
            print xtrain, ytrain, 'xy'
            print aggregated, activated, 'za'
            print delta
        # Perform gradient update for each set of parameters
        for l in range(L-1):
            weights[l] = weights[l] - learning_rate*np.dot(activated[l-1],delta[l].T)
            offsets[l] = offsets[l] - learning_rate*delta[l]
        # Test for convergence
        if 0:
            print weights
            assert False
        acc = classify_accuracy(Xval, Yval, weights, offsets)
        if acc > max_acc:
            print acc
            max_acc = acc
            best_weights = np.copy(weights)
            best_offsets = np.copy(offsets)

            
        if acc <= sum(accuracies[-1*history:])/len(accuracies[-1*history:]):
            return best_weights, best_offsets, max_acc, num_iters
            
        accuracies.append(acc)
        if num_iters%1000==0:
            print 'Iters, acc', num_iters, acc

    
    
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
        if np.isnan(correct):
            print y, predict_y
            assert False
    acc = correct/n1
    if np.isnan(acc):
        return 0
    return acc

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
test_toy = False
if test_toy:
    toy_data = './hw3_resources/data/data_3class.csv'
    train = np.loadtxt(toy_data)
    X = train[:,0:2]
    Y = train[:,2:3].astype(int)
    Xtrain = X[:400,:]
    Ytrain = one_hot(Y[:400,:].reshape(1,-1)[0],3)

    Xval = X[400:600,:]
    Yval = one_hot(Y[400:600,:].reshape(1,-1)[0],3)
    
    Xtest = X[600:,:]
    Ytest = one_hot(Y[600:,:].reshape(1,-1)[0],3)
    print 'Training...'
    weights, offsets , acc, num_iters = NN_train(Xtrain, Ytrain, Xval, Yval, L=4,M=[5,10,3])   
    print 'Finished training in ', num_iters, ' rounds with a validation accuracy of ', acc
    print 'Performance on test set: ', classify_accuracy(Xtest, Ytest, weights, offsets)
    
    def predictNN(x):
        y_vector = NN_predict(x, weights, offsets)
        index = np.nonzero(y_vector)
        return index[0][0]
    # plot validation results
    plot.plotDecisionBoundary(X, Y, predictNN, [-1,0,1], title = 'NN toy set')
    pl.show()
    
#### TEST ON HW2 DATA SETS ####
hw2_data = 0
if hw2_data:
    # parameters
    name = '4'
    print '====== HW2 DATA SET ======'
    # load data from csv files
    train = np.loadtxt('data/data'+name+'_train.csv')
    Xtrain = train[:, 0:2]
    Ytrain_values=train[:, 2:3].astype(int)
    Ytrain_values[Ytrain_values < 0] = 0
    Ytrain = one_hot(Ytrain_values.reshape(1,-1)[0],2)
    val = np.loadtxt('data/data'+name+'_validate.csv')
    Xval = val[:, 0:2]
    Yval_values = val[:, 2:3].astype(int)
    Yval_values[Yval_values < 0] = 0
    Yval = one_hot(Yval_values.reshape(1,-1)[0],2)
    test = np.loadtxt('data/data'+name+'_test.csv')
    Xtest = test[:, 0:2]
    Ytest_values = test[:, 2:3].astype(int)
    Ytest_values[Ytest_values < 0] = 0
    Ytest = one_hot(Ytest_values.reshape(1,-1)[0],2)
    
    print Ytrain_values[:10,:]
    print Ytrain[:10,:]   
    print 'Training...'
    weights, offsets , acc, num_iters = NN_train(Xtrain, Ytrain, Xval, Yval, initial_rate=0.1, L=3, M=[5,2], k=2)   
    print 'Finished training in ', num_iters, ' rounds with a validation accuracy of ', acc
    print 'Performance on test set: ', classify_accuracy(Xtest, Ytest, weights, offsets)
    print 'Performance on training set: ', classify_accuracy(Xtrain, Ytrain, weights, offsets)
    def predictNN(x):
        aggregated, activated = forward_prop(x, weights, offsets)
        y = activated[-1]
        # calculate the expected value to predict for smooth plotting
        return y[1] - y[0]

    # plot validation results
    print Xtest.shape, Ytest_values.shape, 'bye'
    plot.plotDecisionBoundary(Xtest, Ytest_values, predictNN, [0], title = 'Data Set '+name+' using 1 hidden layer with 5 neurons')
    pl.show()
#### TEST ON MNIST DATASETS ####
mnist = 1
normalize = True
if mnist:
    digits = [0,1,2,3,4,5,6,7,8,9]
    train = 20
    val = 5
    test = 50
    Xtrain = np.ndarray((0,784))
    Xval = np.ndarray((0,784))
    Xtest = np.ndarray((0,784))
    Ytrain = np.ndarray((0,10))
    Yval = np.ndarray((0,10))
    Ytest = np.ndarray((0,10))
    Ytest_values = np.ndarray((0,1))
    
    print '====== MNIST DATA SET ======'
    for digit in digits:    
        data = np.loadtxt('data/mnist_digit_'+str(digit)+'.csv')

        X = data[:train+val+test, :]
        Y = np.array([digit]*(train+val+test))
        # normalize data
        if normalize:
            X = 2*X/255 - 1
        Xtrain = np.vstack((Xtrain,X[:train,:]))
        Ytrain = np.vstack((Ytrain,one_hot(Y[:train].reshape(1,-1)[0],10)))
        Xval = np.vstack((Xval, X[train:train+val,:]))
        Yval = np.vstack((Yval,one_hot(Y[train:train+val].reshape(1,-1)[0],10)))
        
        Xtest = np.vstack((Xtest,X[train+val:train+val+test,:]))
        Ytest = np.vstack((Ytest,one_hot(Y[train+val:train+val+test].reshape(1,-1)[0],10)))
        Ytest_values = np.vstack((Ytest_values,Y[train+val:train+val+test].reshape(-1,1)))
    print 'Loaded data'
    print Ytest.shape, Xtest.shape
    print Ytest[:10], Xtest[:10]
    

    print 'Training...'
    print Ytest_values.shape
    weights, offsets , acc, num_iters = NN_train(Xtrain, Ytrain, Xval, Yval, L=4, initial_rate=0.005, fixed=True, M=[120,30,2], k=10)   
    print 'Finished training in ', num_iters, ' rounds with a validation accuracy of ', acc
    print 'Performance on test set: ', classify_accuracy(Xtest, Ytest, weights, offsets)
    print 'Performance on training set: ', classify_accuracy(Xtrain, Ytrain, weights, offsets)

    from matplotlib import pyplot as plt
    
    def visualise_bad_images(Xtest,Ytest):
        #Check dimensions for sanity
        n1, d = Xtest.shape
        n2, k = Ytest.shape
        assert n1==n2
        correct = 0.0
        for i in range(len(X)):
            predict_y = NN_predict(X[i], weights, offsets).reshape(1,-1)
            y = Ytest[i].reshape(1,-1)
            if np.dot(y, predict_y.T)[0][0] < 1:
                x = Xtest[i].reshape(28,28) + 1
                x *= 0.5
                print x
                imgplot = plt.imshow(x)
                imgplot.show()
    visualise_bad_images(Xtest,Ytest)
            


#    Ytrain = one_hot(Ytrain_values.reshape(1,-1)[0],10)
#    print Ytrain_values[:10,:]
#    print Ytrain[:10,:]
#    