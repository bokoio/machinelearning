#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Regression Template
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the DataSet
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Spliting the Datase into The Training set and Test set:
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

"""
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #primeiro transformar depois fazer o fit
X_test = sc_X.transform(X_test)#aqui nao precisa fazer o fit somente transformar
"""

# Fitting Regression Model to the Training set
# Create yor regressor here

# Predicting a new result with Linear Regression

# Predicting a new result with Polynomial Regression
for_pred = np.array(6.5).reshape(1,-1)
y_pred = regressor.predict(for_pred)

# Visualising the Regression Results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()