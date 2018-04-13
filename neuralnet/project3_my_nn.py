from itertools import product
from random import seed
from random import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import math
import cv2

def sigmoids(x, deriv):
    if deriv:
        return x * (1 - x)
    else:
        return float(1)/(1 + np.exp(-x))


#assign initial weights and values to neural network
def initialize(network_architecture, alpha, epochs):
    model = {}
    model['alpha'] = alpha
    model['epochs'] = epochs
    
    weights = []
    for indx, a_layer in enumerate(network_architecture):
        if indx == len(network_architecture) - 1:
            break
        else:
            nextHiddenWeights = np.random.uniform(low = -1, high = 1, size = (network_architecture[indx], network_architecture[indx+1]))
            weights.append(nextHiddenWeights)
    model['weights'] = weights
    return model

#returns a list that is that calculates the error
def find_weights(model, outputs, real):
    output_error = (real - outputs[-1])
    change = sigmoids(outputs[-1], True) * output_error
    changes = []

    reversedOutputs = (list(reversed(outputs)))[1:]
    reversedWeights = list(reversed(model['weights']))

    for indx, output in enumerate(reversedOutputs):
        weight = reversedWeights[indx]
        changes.append(change)
        change = sigmoids(output, True) * np.dot(change, weight.T)

    return list(reversed(changes))

def back_propogate(model, outputs, real):
    #reverse list to deal with last outputs first and backpropogate this way
    updates = find_weights(model, outputs, real)
    updated_weights = []

    #for each layer, multiply output by the weight, and then by alpha + current weight
    for indx, weight in enumerate(model['weights']):
        output = outputs[indx]
        update = updates[indx]
        
        lengthOutputs = len(output)
        lengthWeights = len(update)
        weightMatrix = np.empty(shape = (lengthOutputs, lengthWeights))
        for i in range(lengthOutputs):
            for j in range(lengthWeights):
                weightMatrix[i, j] = output[i]*update[j]
        
        updated_weight = weight + model['alpha'] * weightMatrix
        updated_weights.append(updated_weight)

    return updated_weights

def feed_forward(model, current_inputs):
    #first part of output array is the inputs
    outputs = [current_inputs]
    #add sigmoid of the dot product of each input and the appropriate weights
    #for each subsequent layer
    for w in model['weights']:
        current_inputs = sigmoids(np.dot(current_inputs, w), False)
        outputs.append(current_inputs)
    return outputs

def train(model, training_data):
    for i in range(model['epochs']):
        print i
        correctTrain = 0
        for data in training_data:
            X = data[0]
            y = data[1]            
            outputs = feed_forward(model, np.array(X))
            model['weights'] = back_propogate(model, outputs, np.array(y))
            thisOutput = list(outputs[-1])
            if thisOutput.index(max(thisOutput)) == y.index(max(y)):
                correctTrain += 1

           
        print correctTrain


def predict(model, x):
    return feed_forward(model, x)[-1]

'''
dataset = [([0, 0], [1,0]), ([0,1],[0,1]), ([1,0], [0,1]), ([1,1], [1,0])]
nn = initialize([2, 3, 2], 0.3, 10)
train(nn, dataset)

for (X, y) in dataset:
    print predict(nn, X)
    print y
'''

def prepData():

    print 'prepping data'
    seed(1)
    trainX = np.load('data/tinyX.npy') 
    trainY = np.load('data/tinyY.npy') 


    unique, counts = np.unique(trainY, return_counts=True)

    #print np.asarray((unique, counts)).T


    X_train_throwaway, X_train, y_train_throwaway, y_train = train_test_split(trainX, trainY, test_size=0.25, random_state=42, stratify=trainY)

    unique, counts = np.unique(y_train, return_counts=True)

    maxOfEach = len(X_train) / 10
    dataset = []
    sift = cv2.SIFT(1)


    for indx, im in enumerate(X_train):
        numThisValue = sum(1 for i in dataset if i[-1] == y_train[indx])
        if numThisValue == maxOfEach:
           continue
        newIm = im.transpose(2,1,0)
        #gray= cv2.cvtColor(newIm,cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(newIm, None)
        if len(kp) < 1:
            continue
        
        flat = (desc[0:1].flatten()).tolist()
        #flat = gray.flatten().tolist()
        flat = flat + [y_train[indx]]
        dataset.append(flat)

    #normalize data
    x = np.array(dataset)
    y_values = x[:,-1].astype(float)
    dataset = x / x.max(axis=0)
    dataset[:,-1] = y_values.astype(int)


    dataset = dataset.tolist()
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))


    X = [x[0:-1] for x in dataset]
    Y = [[0 for i in range(n_outputs)] for x in dataset]

    finalY = []
    for indx, expected in enumerate(Y):
        realExpected = dataset[indx][-1]
        expected[int(realExpected)] = 1
        finalY.append(expected)

    Y = finalY

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=42, stratify=Y)

    datasetTrain = zip(X_train, Y_train)
    datasetTest = zip(X_test, Y_test)


    n_inputs = len(dataset[0]) - 1

    print 'n-inputs: ' + str(n_inputs)
    print 'n-outputs: ' + str(n_outputs)
    print 'length dataset: ' + str(len(datasetTrain))


    return n_inputs, n_outputs, datasetTest, datasetTrain
'''
learning_rate = 0.25
epochs = 100

print 'starting training'
nn = initialize([n_inputs, 100, 60, n_outputs], learning_rate, epochs)
train(nn, datasetTrain)

correct = 0
real = []
predictions = []
for (X, y) in datasetTest:
    outputs = predict(nn, X).tolist()
    if outputs.index(max(outputs)) == y.index(max(y)):
        correct += 1

    real.append(y.index(max(y)))
    predictions.append(outputs.index(max(outputs)))

print len(datasetTest)
print correct
print float(correct) / len(datasetTest)

correctTrain = 0
for (X, y) in datasetTrain:
    outputs = predict(nn, X).tolist()
    if outputs.index(max(outputs)) == y.index(max(y)):
        correctTrain += 1


print len(datasetTrain)
print correctTrain
print float(correctTrain) / len(datasetTrain)
'''

