#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Data Processin
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the DataSet
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Spliting the Datase into The Training set and Test set:
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=1/3, random_state = 0)

#Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #primeiro transformar depois fazer o fit
X_test = sc_X.transform(X_test)#aqui nao precisa fazer o fit somente transformar
"""
#Fitting Simple Linear Regrssion to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)










