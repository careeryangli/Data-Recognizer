def groupData(X):
	# Group Training Data into Training, Cross Validation, and Test subsets.
	# Import necessary modules.
	# X is the total dataset.
	# Subset of train_X, train_Y, cv_X, cv_Y, test_X, test_Y are returned.
	import math
	import random
	import numpy as np
	import matplotlib.pyplot as plt

	# Determine the size of each of subsets. 
	# Here 40% data are for training, 30% data are for cross validation
	# 30% data are for test.
	l = len(X)
	train_size = int(math.ceil(l*0.4))
	cv_size = int(math.ceil(l*0.3))
	test_size = l-train_size-cv_size

	# Random shuffle the data.
	random.shuffle(X)

	train_X = []
	train_Y = []
	cv_X = []
	cv_Y = []
	test_X = []
	test_Y = []

	[train_X.append(i[1:]) for i in X[:train_size]]
	[train_Y.append(int(i[0])) for i in X[:train_size]]


	[cv_X.append(i[1:]) for i in X[train_size:train_size+cv_size]]
	[cv_Y.append(int(i[0])) for i in X[train_size:train_size+cv_size]]


	[test_X.append(i[1:]) for i in X[train_size+cv_size:train_size+cv_size+test_size]]
	[test_Y.append(int(i[0])) for i in X[train_size+cv_size:train_size+cv_size+test_size]]

	print 'Total Data Size is %d ...' %len(X)
	print 'Total Training Size is %d ...' %len(train_X)
	print 'Total Cross Validation Size is %d ...' %len(cv_X)
	print 'Total Test Size is %d ...' %len(test_X)

	return train_X, train_Y, cv_X, cv_Y, test_X, test_Y


def MLPnn(X, Y):
	# Neural networ training using Multiple Layer Perceptron.
	# Import necessary modules.
	# X is the train X
	# Y is the train Y

	from sklearn.neural_network import MLPClassifier
	import numpy as np

	print 'Start to train Neural Network...'
	clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (1000, 1000, 1000), random_state = 1)
	print 'Start to fit data...'

	clf.fit(X, Y)
	print clf
	return clf

def accu(Y, predict_Y):
	# Calculate accuracy of predict_Y
	# Y is sample result
	# predict_Y is result predicted by nn
	acc = [1 for i in range(len(Y)) if Y[i] == predict_Y[i]]
	accuracy = len(acc)/float(len(Y))
	return accuracy