""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """
    y = list()
    for row in data:
        z = linear(weights[:-1], row, weights[-1])
        pred0 = 1 / ( 1+ np.exp(-z))
        y.append(pred0)
    # TODO: Finish this function 
    return np.array(y)

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of binary targets. Values should be either 0 or 1
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy.  CE(p, q) = E_p[-log q].  Here
                       we want to compute CE(targets, y).
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    ce = 0
    correct = 0.0
    for i in range(len(targets)):
        if y[i] > 0.5:
            guess = 0
        else:
            guess = 1

        if targets[i] == guess:
            correct+=1.0
        ce_cur = np.log(y[i]) * targets[i]
        ce += ce_cur
    ce = -1 * ce
    return ce, correct/len(targets)

def linear(weights, inputs, bias):
    return np.dot(np.transpose(weights), inputs) + bias

def dfj(targets, weights, inputs, weight_index):
    dfj = 0
    for i in range(len(targets)):
        z = linear(weights[:-1], inputs[i], weights[-1])
        dfj += targets[i] * inputs[i][weight_index] - inputs[i][weight_index] * (np.exp(-z) / (1 + np.exp(-z)))

    return dfj

def dfbias(targets, weights, inputs):
    dfb = 0
    for i in range(len(targets)):
        z = linear(weights[:-1], inputs[i], weights[-1])
        dfb += targets[i] - 1 * (np.exp(-z) / (1 + np.exp(-z)))

    return dfb

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    # TODO: Finish this function

    #bias_vector = np.array(weights[-1] * len(weights) - 1)
    z= list()
    for row in data:
        z.append(np.dot(weights[:-1].transpose(), row))
    
    Z = np.array(z) + weights[-1]
    f = np.dot(targets.transpose(), Z) + np.sum(np.log(1 + np.exp(-Z)))
    
    '''
    for i in range(len(targets)):
        z = linear(weights[:-1], data[i], weights[-1])
        L = 1 + np.exp(-z)
        tz = tz + (targets[i] * z)
        nextf = targets[i] * z + np.log(L)
        f = f + nextf'''
    #iterate over data to find df
    '''T = np.array(targets)
    t = np.array(targets)
    for i in range(len(targets) - 1):
        T = np.hstack((T,t))'''

    df = list()
    for j in range(len(weights) - 1):
            df.append(dfj(targets, weights, data, j))

    df.append(dfbias(targets, weights, data))

    #iterate over data to find y
    return f, np.array(df), logistic_predict(weights, data)


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """

    # TODO: Finish this function

    return f, df, y
