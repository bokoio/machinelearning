# Hierarchical Clustering
#Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5] #4 = Annual.Income 5 = Spending.Score

# Using the Dedograms to find the optimal number of clusters:
dendrogam = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendrogam,
     main = paste('Dendrogam'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

#Fitting HC to the mall dataset
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

#Visualising HC on the clusters
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Cluster of Clients'),
         xlab = "Annual Income",
         ylab = "Spending Score" )