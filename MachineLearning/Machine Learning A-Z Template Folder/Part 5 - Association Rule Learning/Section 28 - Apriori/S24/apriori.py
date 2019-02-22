#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Apriori

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the DataSet
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#Training Aprior on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2,min_lift = 3, min_length = 2)
#para calcular o valor de min_support
#3= a quantidade do produto vendido por dia multiplicado por 7 = numero de dias da semana dividido 7500 que 'e o numero de transaÇoes que temos no banco.
#3*7/7500 = 0.0028
# min_confidence = 0.2 = 20% 

# Visualising the results
results = list(rules)

results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))
results_list_1 = []    
for i in range(0, len(results)):
    results_list_1.append([str(results[i][0]),
                        str(results[i][1]),
                        str(results[i][2][0][2]),
                        str(results[i][2][0][3])])
results_list_1 = pd.DataFrame(data=results_list_1,columns=['RULE','SUPPORT','CONFIDENCE','LIFT'])