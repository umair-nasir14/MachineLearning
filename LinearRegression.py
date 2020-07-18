from __future__ import absolute_import, division, print_function,unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

#Loading dataset

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #training set
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') #testing set
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')


#Getting unique values from feature columns
catagotrical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck','embark_town', 'alone']
numeric_columns = ['age', 'fare']

feature_columns = []
for feature_name in catagotrical_columns:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in numeric_columns:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_inputfunction(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def inputfunction():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
        return ds  # return a batch of the dataset
    return inputfunction  # return a function object for use

#making input functions
train_if = make_inputfunction(dftrain,y_train)
eval_if = make_inputfunction(dfeval,y_eval,num_epochs=1,shuffle=False)
#training and testing models
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_if) 
result = linear_est.evaluate(eval_if) 

#predicting survival rate  
result =list(linear_est.predict(eval_if))
print(dfeval.loc[15])
print(result[15]['probabilities'][1]) #here 15 is the persons entry number, 1 represents probability of surviving 0 for not surviving






