import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import collections
import random
import math
import cProfile
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#ensure gamma is never positive
def sigmoid_funct(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    return 1 / (1 + math.exp(-gamma))

#chi function
def chi(weights, i):
	probabilities = 0
	for indx, weight in enumerate(weights):
		probabilities = probabilities + (i[indx] * weight)
	sigmoid = sigmoid_funct(probabilities)
	return sigmoid


#implementing my gradient descent
def gradientDescent(feats, y, weights, step, numOccurences, iterations):
	lastWeights = weights
	amountTime = iterations
	for j in range(0, iterations):
		totalSum = np.zeros(len(feats[0]))
		for indx, i in enumerate(feats):
			sigmoid = chi(weights, i)
			real = y[indx]
			loss = real - sigmoid
			newWeights = loss * i
			totalSum = totalSum + newWeights
		weights = weights + (step * totalSum)
		print weights
		if np.array_equal(np.around(weights, decimals=7), np.around(lastWeights, decimals=7)):
			amountTime = j
			break
		lastWeights = weights
	print 'final weights'
	print weights
	return weights, amountTime

########################## TESTING COMPLEXITY #######################
#testing what happens when I increase the number of features used. This is tested through using each combination 
#of years
#Generate Graphs. 
def sample_Complexity(X_train, y_train, X_dev, y_dev):

	raceYears = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']

	recallScores = list()
	complexityList = list()
	precisionScores = list()
	accuracyScores = list()
	fScores = list()

	recallScoresTrain = list()
	precisionScoresTrain = list()
	accuracyScoresTrain = list()
	fScoresTrain = list()
	indxArray = range(len(raceYears))


	#increment by year and add one more feature for that year, sum the rest of the races
	for i, indx in enumerate(range(len(raceYears))):
		i = i + 1
		year_feats = indxArray[-i:]
		years_to_add = indxArray[0:len(raceYears) - i]

		#format array
		feat_array_train = X_train[:,year_feats]
		to_sum_array_train = X_train[:,years_to_add]
		sum_array_train = np.sum(to_sum_array_train, axis=1)
		sum_array_train = np.reshape(sum_array_train, (len(sum_array_train), 1))
		finalFeats_train = np.concatenate((feat_array_train, sum_array_train), axis=1)

		occurences_train,numWeights_train, train = add_column(finalFeats_train)

		feat_array_dev = X_dev[:,year_feats]
		to_sum_array_dev = X_dev[:,years_to_add]
		sum_array_dev = np.sum(to_sum_array_dev, axis=1)
		sum_array_dev = np.reshape(sum_array_dev, (len(sum_array_dev), 1))
		finalFeats_dev = np.concatenate((feat_array_dev, sum_array_dev), axis=1)

		occurences_dev,numWeights_dev, dev = add_column(finalFeats_dev)


		maxIterations = 600
		alpha = 0.001

		#run the gradient descent
		weights = np.ones(numWeights_train)
		weights1, numIterations1 = gradientDescent(train, y_train, weights, alpha, occurences_train, maxIterations)
		
		#test the scores
		precision1, recall1, accuracy1 = getScores(dev, y_dev, weights1)
		precision_train1, recall_train1, accuracy_train1 = getScores(train, y_train, weights1)

		fscore = float(2 * (precision1 * recall1)) / (precision1 + recall1)
		fscoreTrain = float(2 * (precision_train1 * recall_train1)) / (precision_train1 + recall_train1)


		recallScores.append(recall1)
		precisionScores.append(precision1)
		accuracyScores.append(accuracy1)
		complexityList.append(numWeights_train)
		fScores.append(fscore)

		recallScoresTrain.append(recall_train1)
		precisionScoresTrain.append(precision_train1)
		accuracyScoresTrain.append(accuracy_train1)
		fScoresTrain.append(fscoreTrain)
		print recallScores
		print recallScoresTrain
		print precisionScores
		print precisionScoresTrain
		print accuracyScores
		print fScores
		print fScoresTrain

	print recallScoresTrain
	print precisionScoresTrain
	print accuracyScoresTrain
	print fScoresTrain 

	print recallScores
	print precisionScores
	print accuracyScores
	print fScores

	print recallScoresTrain
	print precisionScoresTrain
	print accuracyScoresTrain
	print fScoresTrain 

	plt.plot(complexityList, recallScores)
	plt.plot(complexityList, recallScoresTrain)
	plt.legend(['Validation', 'Train'], loc='lower right')
	plt.xlabel('Model Complexity', fontsize=16)
	plt.ylabel('Prediction: Recall')
	plt.savefig('modelComplexity_Recall2.png')
	plt.show()

	plt.plot(complexityList, precisionScores)
	plt.plot(complexityList, precisionScoresTrain)
	plt.legend(['Validation', 'Train'], loc='lower right')
	plt.xlabel('Model Complexity', fontsize=16)
	plt.ylabel('Prediction: Precision')
	plt.savefig('modelComplexity_Precision2.png')
	plt.show()


	plt.plot(complexityList, accuracyScores)
	plt.plot(complexityList, accuracyScoresTrain)
	plt.legend(['Validation', 'Train'], loc='lower right')
	plt.xlabel('Model Complexity', fontsize=16)
	plt.ylabel('Prediction: Accuracy')
	plt.savefig('modelComplexity_Accuracy2.png')
	plt.show()

	plt.plot(complexityList, fScores)
	plt.plot(complexityList, fScoresTrain)
	plt.legend(['Validation FScore', 'Train FScore'], loc='lower right')
	plt.xlabel('Model Complexity', fontsize=16)
	plt.ylabel('Prediction')
	plt.savefig('modelComplexity_fscore2.png')
	plt.show()

	plt.plot(complexityList, accuracyScores)
	plt.plot(complexityList, accuracyScoresTrain)
	plt.plot(complexityList, fScores)
	plt.plot(complexityList, fScoresTrain)
	plt.legend(['Validation Accuracy', 'Train Accuracy', 'Validation FScore', 'Train FScore'], loc='lower right')
	plt.xlabel('Model Complexity', fontsize=16)
	plt.ylabel('Prediction')
	plt.savefig('modelComplexity_Accuracy_fscore2.png')
	plt.show()

########################## TESTING ALPHAS #######################
def alphaTests(X_train, y_train, X_dev, y_dev):
	train = np.concatenate((X_train, X_dev), axis=0)
	ytrain = np.concatenate((y_train, y_dev), axis=0)

	raceYears = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']

	indxArray = range(len(raceYears))
	i = 2
	year_feats = indxArray[-i:]
	print year_feats
	years_to_add = indxArray[0:len(raceYears) - i]
	print years_to_add


	#format the data with the right number of features
	feat_array_train = X_train[:,year_feats]
	to_sum_array_train = X_train[:,years_to_add]
	sum_array_train = np.sum(to_sum_array_train, axis=1)
	sum_array_train = np.reshape(sum_array_train, (len(sum_array_train), 1))
	finalFeats_train = np.concatenate((feat_array_train, sum_array_train), axis=1)

	occurences_train,numWeights_train, train = add_column(finalFeats_train)

	train = finalFeats_train
	trains = np.array_split(train, 5)
	targets = np.array_split(ytrain, 5)
	cross_valAlpha(trains, targets)

#uses cross-validation to determine what the best alpha value is
#divide the training set into 3 partitions and train on two at a time
def cross_valAlpha(trains, targets):
	train1 = np.concatenate((trains[0], trains[1]))
	target1 = np.concatenate((targets[0], targets[1]))

	occurences1,numWeights, train1 = add_column(train1)
	occurences1_test,numWeights, test1 = add_column(trains[2])
	targets_test1 = targets[2]

	train2 =np.concatenate((trains[1], trains[2]))
	target2 = np.concatenate((targets[1], targets[2]))

	occurences2,numWeights, train2 = add_column(train2)
	occurences2_test,numWeights, test2 = add_column(trains[0])
	targets_test2 = targets[0]


	train3 = np.concatenate((trains[0], trains[2]))
	target3 = np.concatenate((targets[0], targets[2]))

	occurences3,numWeights, train3 = add_column(train3)
	occurences3_test,numWeights, test3 = add_column(trains[1])
	targets_test3 = targets[1]


	#generate 10 different alpha scores across 10^-5 to 10^-2.75
	maxIterations= 1000
	all_alphas = np.logspace(-4, -2.5, 10)
	print all_alphas

	recallScores = list()
	alphasList = list()
	precisionScores = list()
	accuracyScores = list()
	fScores = list()
	fScoresTrain = list()
	itersList = list()

	for alpha in all_alphas:
		accuracy = 0
		precision = 0
		recall = 0

		accuracyTrain = 0
		precisionTrain = 0
		recallTrain = 0

		weights = np.ones(numWeights)
		weights1, numIterations1 = gradientDescent(train1, target1, weights, alpha, occurences1, maxIterations)
		precision1, recall1, accuracy1 = getScores(test1, targets_test1, weights1)
		precision_train1, recall_train1, accuracy_train1 = getScores(train1, target1, weights1)

		weights = np.ones(numWeights)
		weights2,numIterations2 = gradientDescent(train2, target2, weights, alpha, occurences2, maxIterations)
		precision2, recall2, accuracy2 = getScores(test2, targets_test2, weights2)
		precision_train2, recall_train2, accuracy_train2 = getScores(train2, target2, weights2)

		weights = np.ones(numWeights)
		weights3,numIterations3 = gradientDescent(train3, target3, weights, alpha, occurences3, maxIterations)
		precision3, recall3, accuracy3 = getScores(test3, targets_test3, weights3)
		precision_train3, recall_train3, accuracy_train3 = getScores(train3, target3, weights3)

		accuracy = float(accuracy1 + accuracy2 + accuracy3) / 3
		precision = float(precision1 + precision2 + precision3) / 3
		recall = float(recall1 + recall2 + recall3) / 3

		accuracyTrain = float(accuracy_train1 + accuracy_train2 + accuracy_train3) / 3
		precisionTrain = float(precision_train1 + precision_train2 + precision_train3) / 3
		recallTrain = float(recall_train1 + recall_train2 + recall_train3) / 3

		numIterations = float(numIterations1 + numIterations2 + numIterations3) / 3

		fscore = (2 * (precision * recall)) / (precision + recall)
		fscoreTrain = (2 * (precisionTrain * recallTrain)) / (precisionTrain + recallTrain)
		print fscore
		print fscoreTrain

		print 'average for alpha = ' + str(alpha)
		fScores.append(fscore)
		fScoresTrain.append(fscoreTrain)
		recallScores.append(recall)
		precisionScores.append(precision)
		accuracyScores.append(accuracy)
		alphasList.append(alpha)
		itersList.append(numIterations)

	plt.plot(all_alphas, recallScores)
	plt.plot(all_alphas, precisionScores)
	plt.plot(all_alphas, accuracyScores)
	plt.legend(['R', 'P', 'Accuracy'], loc='upper right')
	plt.xlabel('Alpha', fontsize=16)
	plt.ylabel('percent')
	plt.savefig('alphasAccuracy1.png')
	plt.show()

	plt.plot(all_alphas, itersList)
	plt.legend('num of Iterations', loc='upper right')
	plt.savefig('alphasIterations1.png')
	plt.show()

	plt.plot(all_alphas, fScores)
	plt.plot(all_alphas, fScoresTrain)
	plt.legend(['Validation', 'Training'], loc='upper right')
	plt.xlabel('Alpha', fontsize=16)
	plt.ylabel('FScore')
	plt.savefig('alphasfscore1.png')
	plt.show()


#method that adds a column of 1s for the bias value and returns the new 
#training array and the shape of the training array
def add_column(X):
	numOccurences, numWeights = np.shape(X)
	ones = np.ones((numOccurences,1))
	X = np.append(X, ones, axis=1)
	numOccurences, numWeights = np.shape(X)

	return numOccurences, numWeights, X

#uses different complexity of features to decide on which model is the best one


#applies the weights to find the prediction
def getPrediction(x, weights):
	probabilities = 0
	for indx, weight in enumerate(weights):
		probabilities = probabilities + (x[indx] * weight)
	#sigmoid = 1 / (1 + math.exp(-probabilities))
	sigmoid = sigmoid_funct(probabilities)
	if sigmoid >= 0.5:
	    return 1
	else:
	    return 0

#predict function for an array of features
def predict(X, weights):
	return([getPrediction(x, weights) for x in X])

#get the precision, accuracy and recall for all predictions
def getScores(X, y, weights):
	total = 0
	correct = 0

	totalOnes = 0
	totalZeros = 0

	realpos = 0
	falsepos = 0
	realneg = 0
	falseneg = 0

	predictions = predict(X, weights)
	precision = 0
	recall = 0
	accuracy = 0
	for indx, prediction in enumerate(predictions):
	    real = y[indx]
	    if prediction == 1 and real == 1:
	        correct += 1
	        realpos += 1
	        totalOnes += 1

	    elif prediction == 0 and real == 0:
	        correct += 1
	        totalZeros += 1
	        realneg += 1

	    elif real == 1:
	        totalOnes += 1
	        falseneg += 1

	    elif real == 0:
	        totalZeros += 1
	        falsepos += 1
	    total += 1

	if realpos != 0:
	    precision = float(realpos) / (realpos + falsepos)
	    recall = float(realpos) / (realpos + falseneg)
	    accuracy = float(correct) / total
	else:
	    precision = 0
	    recall = 0
	    accuracy = float(correct) / total

	print 'all the scores'

	print total
	print totalOnes
	print totalZeros

	print realpos
	print falsepos
	print realneg
	print falseneg

	print precision
	print recall
	print accuracy

	return precision, recall, accuracy

#formats the data into train, test, and validator set 
def deal_with_data(csv):
	data = pd.read_csv(csv)
	df = pd.DataFrame(data)
	df = df.sample(frac=1)
	sample_size = sum(df['2016'] == 1)


	target2016 = pd.DataFrame(df, columns = ['2016'])
	
	dfwith2016 = pd.DataFrame(df, columns = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016'])
	df = pd.DataFrame(df, columns = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'])

	mask = np.all(df == 0, axis=1)
	df = dfwith2016[~mask]

	

	numberLines = len(df)
	twentyPercent = int(numberLines * 0.20)

	test = df[0:twentyPercent]
	val = df[-twentyPercent:]
	df = df[twentyPercent:-twentyPercent]

	sample_size = sum(df['2016'] == 1)

	numNegatives = (sample_size * 95)/5

	negative_indices = df[df['2016'] == 0].index
	positive_indices = df[df['2016'] == 1].index
	np.random.seed(0)
	random_indices = np.random.choice(negative_indices, numNegatives, replace=False)
	
	negatives = df.loc[random_indices]
	positives = df.loc[positive_indices]

	df = pd.concat([negatives, positives])
	df = df.sample(frac=1)

	
	X_train = pd.DataFrame(df, columns = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'])
	y_train = pd.DataFrame(df, columns = ['2016'])
	X_train = X_train.values
	y_train = y_train.values

	X_test = pd.DataFrame(test, columns = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'])
	y_test = pd.DataFrame(test, columns = ['2016'])
	X_test = X_test.values
	y_test = y_test.values

	X_val = pd.DataFrame(val, columns = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'])
	y_val = pd.DataFrame(val, columns = ['2016'])
	X_val = X_val.values
	y_val = y_val.values
	


	print sum(y_train)
	print sum(y_val)
	print sum(y_test)

	return X_train, y_train, X_test, y_test, X_val, y_val

#test my logestic regression on test data
def testSetMeasures(X_train, y_train, X_test, y_test):
	raceYears = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']


	indxArray = range(len(raceYears))
	i = 2
	year_feats = indxArray[-i:]
	years_to_add = indxArray[0:len(raceYears) - i]

	feat_array_train = X_train[:,year_feats]
	to_sum_array_train = X_train[:,years_to_add]
	sum_array_train = np.sum(to_sum_array_train, axis=1)
	sum_array_train = np.reshape(sum_array_train, (len(sum_array_train), 1))
	finalFeats_train = np.concatenate((feat_array_train, sum_array_train), axis=1)

	occurences_train,numWeights_train, train = add_column(finalFeats_train)

	feat_array_test = X_test[:,year_feats]
	to_sum_array_test = X_test[:,years_to_add]
	sum_array_test = np.sum(to_sum_array_test, axis=1)
	sum_array_test = np.reshape(sum_array_test, (len(sum_array_test), 1))
	finalFeats_test = np.concatenate((feat_array_test, sum_array_test), axis=1)

	occurences_test,numWeights_test, test = add_column(finalFeats_test)

	maxIterations = 600
	alpha = 0.001

	weights = np.ones(numWeights_train)
	#weights1, numIterations1 = gradientDescent(train, y_train, weights, alpha, occurences_train, maxIterations)
	weights1 = [ 1.90501845,  2.71082315,  0.37632518, -4.48068862]
	precision1, recall1, accuracy1 = getScores(test, y_test, weights1)
	print 'precision: ' + str(precision1)
	print 'recall: ' + str(recall1)
	print 'accuracy: ' + str(accuracy1)

	print 'final weights'
	print weights1

	fscore = float(2 * (precision1 * recall1)) / (precision1 + recall1)
	return weights1

#use the new, 2015 feats for the real prediction
def organize_data_2017(csv):
	data = pd.read_csv(csv)
	df = pd.DataFrame(data)

	df['2015'] = df['2015'].astype(int)
	df['2016'] = df['2016'].astype(int)
	df['all'] = df['2003'] + df['2004'] + df['2005'] + df['2006'] + df['2007'] + df['2008'] + df['2009'] + df['2010'] + df['2011'] + df['2012'] + df['2013'] + df['2014']


	df_a = pd.DataFrame(df, columns = ['Id', '2015', '2016', 'all'])

	X = df_a.values
	
	return X

#use a set of weights to predict the 2017 data
def getPredictions2017(weights):
	X = organize_data_2017("reformattedData_int.csv")

	numOccurences_2017, numWeights_2017 = np.shape(X)
	ones = np.ones((numOccurences_2017,1))
	X = np.append(X, ones, axis=1)
	numOccurences_2017, numWeights_2017 = np.shape(X)

	numIterations= 700
	alpha = 0.001


	print "2017 predictions"

	predictions = np.zeros(numOccurences_2017)
	listRunners = []
	for indx, feats in enumerate(X):
		runnerID = feats[0]
		probabilities = (feats[1] * weights[0]) + (feats[2] * weights[1]) + (feats[3] * weights[2]) + (feats[4] * weights[3])
		sigmoid = 1 / (1 + math.exp(-probabilities))
		if sigmoid >= 0.5:
			predictions[indx] = 1
			listRunners.append([runnerID, 1])
		else:
			predictions[indx] = 0
			listRunners.append([runnerID, 0])

	print len(listRunners)

	f = open('arianePredictionsLogReg.csv', 'w')
	for row in listRunners:
		f.write(str(row[0]) + ',' + str(row[1]) + '\n')  
	f.close()


	print predictions
	print np.count_nonzero(predictions)

	

def main():
	csvfile = "reformattedData_int.csv"
	X_train, y_train, X_test, y_test, X_val, y_val = deal_with_data(csvfile)
	print "Length Train: " + str(len(X_train))
	print "Length Test: " + str(len(X_test))
	print "Length Val: " + str(len(X_val))

	#different testing mechanisms
	#alphaTests(X_train, y_train, X_val, y_val)
	#weights = sample_Complexity(X_train, y_train, X_val, y_val)


	alpha = 0.001
	weights1 = testSetMeasures(X_train, y_train, X_test, y_test)
	weights1 = [ 1.90501845,  2.71082315,  0.37632518, -4.48068862]
	getPredictions2017(weights1)

		



if __name__ == '__main__':
	main()

