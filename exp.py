#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: expert
"""

#IMPORTING THE REQUIRED LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

#LOADING THE DATASET
df=pd.read_csv("/home/expert/Downloads/FORSKS TECHNOLOGIES/Forsk_Session_Dataset/Foodtruck.csv")

#TOP 5 ROWS 
df.head(5)

#PERFORMING EXPLORATORY DATA ANALYSIS
#ROWS AND COLUMNS
df.shape

#COLUMN NAMES 
df.columns.tolist()

#CHECKING MISSING VALUES
df.isnull().any(axis=0)

#NULL VALUES SUM
df.isnull().sum()

#INFO ON DATAFRAME
df.info()

#STATS OF DATAFRAME
df.describe()

#TOP 5 ROWS OF POPULATION AND PROFIT
df["Population"].head()
df["Profit"].head()

#CHECKING ON INDIVIDUAL COLUMNS
df["Population"]
df["Profit"]

#FEATURES 
features=df.iloc[:,0:1].values
features

#CHECKING TYPE OF FEATURES
type(features)

#FEATURES DIMENSION
features.ndim

#LABELS
labels=df.iloc[:,1:2].values
labels


#SPLITTING THE DATASET INTO TRAIN_TEST_SPLIT TTS AS TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split as TTS
features_train,features_test,labels_train,labels_test=TTS(features,labels,test_size=0.2,random_state=0)

#FEATURES TRAIN 
features_train

#FEATURES
features

#FEATURES TEST 
features_test


#LABELS TRAIN
labels_train

#LABELS
labels

#LABELS TEST
labels_test

#FEATURES TRAIN SHAPE
features_train.shape

#FEATURES TEST SHAPE
features_test.shape

#LABELS TRAIN SHAPE
labels_train.shape

#LABELS TEST SHAPE
labels_test.shape

#Plotting Graphs SCATTER PLOT
plt.scatter(features,labels)
plt.title("Population Vs Profit Plot")
plt.xlabel("Population", features)
plt.ylabel("Profit",labels)
plt.show()

#HIST PLOT
sb.histplot(features)
sb.histplot(labels)

#FITTING LINEAR REGRESSION MODEL ON TRAINING DATASET
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()  

#RESHAPING FEATURES_TRAIN
features_train=features_train.reshape(len(features_train),1)

#CHECKING THE DIMENSIONS
features_train.ndim
regressor.fit(features_train, labels_train)

#CHECKING THE ML ALGORITHM
regressor

#LENGTH OF FEATURES TRAIN
len(features_train)

#FEATURES TRAIN ROWS AND COLUMNS
features_train.shape

#LABELS TRAIN ROWS AND COLUMNS
labels_train.shape

#FEATURES TRAIN DIMENSIONS
features_train.ndim

#LABELS TRAIN DIMENSIONS
labels_train.ndim

#FEATURES TEST DIMENSIONS
features_test.ndim

#MAKING PREDICTIONS
regressor.predict([[33.4]])


#EVALUATING THE MODEL USING A NEW INPUT VARIABLE
x=[3.073]

#CHECKING THE TYPE OF THE INPUT
type(x)

#CHANGING LIST INTO ARRAY
x=np.array(x)
type(x)

#CHECKING INPUT DIMENSION
x.ndim

#CHECKING INPUT ROWS AND COLUMNS
x.shape

#RESHAPING THE INPUT VALUE FROM 1d TO 2d
x=x.reshape(1,1)
x.ndim

#PREDICTING THE INPUT VALUE
regressor.predict(x)

#IMORTING PICKLE FILE
import pickle

#DUMPING THE PICKLE FILE
filename="store.sav"
pickle.dump(regressor,open(filename,"wb"))

#LOADING THE PICKLE FILE
filename="store.sav"
loaded_model=pickle.load(open(filename,"rb"))
