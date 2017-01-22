from DataProcess import *
from ML import *
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

count = 1
predict = []
model = []
predict_result = []

X = Reg_loadCSV('/Users/yangli/Documents/Programming/Python Project/Kaggle/Digit Recognizer/Data/train.csv')

print np.shape(X)
train_X, train_Y, cv_X, cv_Y, test_X, test_Y = groupData(X)

clf = MLPnn(train_X[:], train_Y[:])
print 'Training is done...'

cv_predict = [clf.predict(np.reshape(i, (1, -1))).tolist() for i in cv_X]
cv_predict = np.reshape(cv_predict, [1, len(cv_predict)]).tolist()
cv_predict = cv_predict[0]
print 'Prediction is done'

print len(cv_predict)

accuracy = accu(cv_Y[:], cv_predict)
print cv_predict[:100]
print cv_Y[:100]
print accuracy
