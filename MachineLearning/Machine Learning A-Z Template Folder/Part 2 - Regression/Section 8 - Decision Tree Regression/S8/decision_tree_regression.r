#Decision Tree Regression
# Regression Template ~
#Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]


#Spliting the Datase into The Training set and Test set:

#library(caTools)
#set.seed(123)
#split = sample.split(dataset$Purchased, SplitRatio = 0.8)
#training_set = subset(dataset, split == TRUE)
#test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Decision Tree Regression to the Dataset
# Create Regressor here
#install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

# Predicting a new result.
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Regression Model Results.
#install.packages('ggplot2')
library(ggplot2)
ggplot()+
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level , y = predict(regressor, newdata = dataset)),
            color = 'blue')+
  ggtitle('Truth or Bluff(Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Regression  Model Results (for higher resolution and smoother curve).
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')
plot(regressor)
text(regressor)