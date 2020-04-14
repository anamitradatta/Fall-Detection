# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.
"""

import os
import sys
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle
import sklearn


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'my-activity-data.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------
'''
print("Reorienting data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,2], data[i,3], data[i,4]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)
'''
#print(data)
accel_data = np.asarray([[data[i,0],data[i,1],data[i,2],data[i,3],data[i,7]] for i in range(len(data))])
gyro_data = np.asarray([[data[i,0],data[i,4],data[i,5],data[i,6],data[i,7]] for i in range(len(data))])
print(accel_data)
#print(gyro_data)
# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 200
step_size = 5

# sampling rate should be about 25 Hz; you can take a brief window to confirm this
'''
n_samples = 1000
time_elapsed_seconds = (accel_data[n_samples,0] - accel_data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds
'''
# TODO: list the class labels that you collected data for in the order of label_index (defined in collect-labelled-data.py)
class_names = ["falling", "jumping", "sitting", "standing","turning","walking"] #...

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []

for i,window_with_timestamp_and_label in slidingWindow(accel_data, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]   
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[100, -1])
   
for i,window_with_timestamp_and_label in slidingWindow(gyro_data, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]   
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[100, -1])
    
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

# TODO: split data into train and test datasets using 10-fold cross validation

cv = sklearn.model_selection.KFold(n_splits=10, random_state=None, shuffle=True)

"""
TODO: iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""

accuracy = np.zeros(10)
counter = 0

for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    #tree = sklearn.tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
    tree = RandomForestClassifier(n_estimators=100)
    #tree = BaggingClassifier(n_estimators=100)
    #tree = ExtraTreesClassifier(n_estimators=100)
    #tree = GradientBoostingClassifier(n_estimators=100)
    #tree = GaussianNB()
    tree.fit(X_train,Y_train)
    Y_pred = tree.predict(X_test)
    conf = sklearn.metrics.confusion_matrix(Y_test, Y_pred)


# TODO: calculate and print the average accuracy, precision and recall values over all 10 folds
    print(conf)
    print('Accuracy Score :',sklearn.metrics.accuracy_score(Y_test, Y_pred))
    accuracy[counter] = sklearn.metrics.accuracy_score(Y_test, Y_pred)
    counter+=1
    print('Report : ')
    print(sklearn.metrics.classification_report(Y_test, Y_pred))
    
print("average accuracy = " + str(np.mean(accuracy)))

# TODO: train the decision tree classifier on entire dataset
#activity_classifier = sklearn.tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
activity_classifier = RandomForestClassifier(n_estimators=100)
#activity_classifier = BaggingClassifier(n_estimators=100)
#activity_classifier = ExtraTreesClassifier(n_estimators=100)
#activity_classifier = GradientBoostingClassifier(n_estimators=100)
#activity_classifier = GaussianNB()
activity_classifier = activity_classifier.fit(X,Y) 

# TODO: Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line
    
#export_graphviz(activity_classifier, out_file='tree.dot', feature_names = feature_names)
    
# TODO: Save the classifier to disk - replace 'tree' with your decision tree and run the below line
with open('classifier.pickle', 'wb') as f:
    pickle.dump(activity_classifier, f)
  