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

    # TODO: Finish this function

    return y

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
    
    return ce, frac_correct

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

    #Iterate over the data to find f
    f = 0
    for i in range(len(targets)):
        z = linear(weights[:-1], data[i], weights[-1])
        L = 1 + np.exp(-z)
        nextf = targets[i] * z + np.log(L)
        f = f + nextf
    
    #iterate over data to find df
    df = list()
    for j in range(len(weights) - 1):
            df.append(dfj(targets, weights, data, j))

    df.append(dfbias(targets, weights, data))

    #iterate over data to find y
    y = list()
    for i in range(len(targets)):
        z = linear(weights[:-1], data[i], weights[-1])
        pred0 = 1 / ( 1+ np.exp(-z))
        if(targets[i] == 1):
            y.append(1 - pred0)
        else:
            y.append(pred0)

    return f, np.array(df), np.array(y)


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
