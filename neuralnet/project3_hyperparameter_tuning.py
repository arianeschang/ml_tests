import project3_my_nn as nn
from operator import itemgetter
import pickle
from sklearn.metrics import confusion_matrix, classification_report


def runModel(layerArchi, learning_rate, epochs, datasetTest, datasetTrain):

	#print 'hyperparameters: # layers: ' + str(len(layerArchi)) + ', size of layers: ' + str(layerArchi) + ', alpha: ' + str(learning_rate) + ', epochs: ' + str(epochs)
	model = nn.initialize(layerArchi, learning_rate, epochs)
	nn.train(model, datasetTrain)

	correct = 0
	real = []
	predictions = []
	for (X, y) in datasetTest:
	    outputs = nn.predict(model, X).tolist()
	    if outputs.index(max(outputs)) == y.index(max(y)):
	        correct += 1

	    real.append(y.index(max(y)))
	    predictions.append(outputs.index(max(outputs)))

	accuracyTest = float(correct) / len(datasetTest)

	correctTrain = 0
	real = []
	predictions = []

	for (X, y) in datasetTrain:
		outputs = nn.predict(model, X).tolist()
		if outputs.index(max(outputs)) == y.index(max(y)):
			correctTrain += 1
		real.append(y.index(max(y)))
		predictions.append(outputs.index(max(outputs)))


	print 'accuracy Test: ' + str(accuracyTest), 'accuracy train: ' + str(float(correctTrain) / len(datasetTrain))
	print classification_report(real, predictions)
	return accuracyTest

def chunkList(lst, n):
	return [ lst[i::n] for i in xrange(n) ]


def main():
	'''
	n_inputs, n_outputs, datasetTest, datasetTrain = nn.prepData()
	pickle.dump(datasetTrain, open('data/datasetTrain', 'wb'))
	pickle.dump(datasetTest, open('data/datasetTest', 'wb'))
	'''
	print 'reading data'
	datasetTrain = pickle.load(open('data/datasetTrain', 'rb'))
	datasetTest = pickle.load(open('data/datasetTest', 'rb'))
	n_inputs = 128
	n_outputs = 40
	print len(datasetTrain)

	'''
	#chunk list for cross validation
	chunkedLists = chunkList(datasetTrain, 3)
	crossValDatasets = []
	for i in range(len(chunkedLists)):
		test_chunk = chunkedLists[i]
		train_chunk = [chunkedLists[indx] for indx in range(len(chunkedLists)) if indx is not i]
		flat_train = [item for sublist in train_chunk for item in sublist]
		crossValDatasets.append((flat_train, test_chunk))

	#parameters to tune
	print 'training'
	
	#parameters to test
	learning_rate_params = [0.1, 0.2, 0.3, 0.4, 0.5]
	epochs_params = [5, 10, 25, 50, 100]
	layerArchiList = [[n_inputs, 10, n_outputs], [n_inputs, 20, n_outputs], [n_inputs, 40, n_outputs], \
					[n_inputs, 50, n_outputs], [n_inputs, 10, 20, n_outputs], [n_inputs, 10, 40, n_outputs], [n_inputs, 10, 50, n_outputs], \
					[n_inputs, 20, 40, n_outputs], [n_inputs, 20, 50, n_outputs], [n_inputs, 40, 50, n_outputs], \
					[n_inputs, 10, 20, 40, n_outputs], [n_inputs, 10, 20, 50, n_outputs], [n_inputs, 10, 40, 50, n_outputs], \
					[n_inputs, 20, 40, 50, n_outputs], [n_inputs, 10, 20, 40, 50, n_outputs]]


	print 'num combos'
	print len(layerArchiList) * len(learning_rate_params) * len(epochs_params)
	print 'num trains'
	print len(layerArchiList) * len(learning_rate_params) * len(epochs_params) * 3
	results = []
	for layerArchi in layerArchiList:
		for rate in learning_rate_params:
			for epochs in epochs_params:
				input_params = 'hyperparameters: # layers: ' + str(len(layerArchi)) + ', size of layers: ' + str(layerArchi) + ', alpha: ' + str(rate) + ', epochs: ' + str(epochs)
				print input_params
				accuracies = []
				for indx, (datasetTrain, datasetTest) in enumerate(crossValDatasets):
					print 'cross-val set: ' + str(indx)
					accuracy = runModel(layerArchi, rate, epochs, datasetTest, datasetTrain)
					accuracies.append(accuracy)

				mean_accuracy = sum(accuracies)/float(len(accuracies))
				print 'mean accuracy: ' + str(mean_accuracy)
				
				results.append((mean_accuracy, input_params))
				print 'current maximum: ' + str(max(results, key=itemgetter(0))[1])
			print 'current maximum: ' + str(max(results, key=itemgetter(0))[1])
		print 'current maximum: ' + str(max(results, key=itemgetter(0))[1])

	maximumAccuracy = max(results, key=itemgetter(0))[1]
	print maximumAccuracy
	print results
	'''

	accuracy = runModel([128, 50, 40], 0.4, 25, datasetTest, datasetTrain)
	print accuracy


if __name__ == '__main__':
   main()