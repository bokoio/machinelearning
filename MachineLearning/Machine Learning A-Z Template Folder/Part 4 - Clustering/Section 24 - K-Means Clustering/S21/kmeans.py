#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 10:12:44 2019

@author: pippo


K-Means Clustering

"""
#reset -f

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
#Array com as colunas usadas:
X = dataset.iloc[:,[3,4]].values

#using the elbow method to find the optimal numbers of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# applying k-means to the mall dataset
kmeans = KMeans(n_clusters=5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the cluster
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1], s=100, c='red',     label = 'Cluster 1' )
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1], s=100, c='blue',    label = 'Cluster 2' )
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1], s=100, c='green',   label = 'Cluster 3' )
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3,1], s=100, c='cyan',    label = 'Cluster 4' )
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4,1], s=100, c='magenta', label = 'Cluster 5' )
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300,c='yellow', label = 'Centrouids')
plt.title('Cluster of clients')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# Visualising the cluster Mesmo codigo de cima mas classificando os Clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1], s=100, c='red',     label = 'Careful' )
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1], s=100, c='blue',    label = 'Standard' )
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1], s=100, c='green',   label = 'Target' )
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3,1], s=100, c='cyan',    label = 'Careless' )
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4,1], s=100, c='magenta', label = 'Sensible' )
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300,c='yellow', label = 'Centrouids')
plt.title('Cluster of clients')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

