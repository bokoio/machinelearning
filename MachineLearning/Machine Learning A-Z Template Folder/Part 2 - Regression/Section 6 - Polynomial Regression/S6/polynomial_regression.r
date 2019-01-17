#Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

library(caTools)
set.seed(123)
#split = sample.split(dataset$Purchased, SplitRatio = 0.8)
#training_set = subset(dataset, split == TRUE)
#test_set = subset(dataset, split == FALSE)

# Fitting Simple Linear Regression to the Dataset
lin_reg = lm(formula = Salary ~ ., data = dataset)

# Fitting Polynomial Regression to the Dataset
dataset$Level2 = dataset$Level^2 #criacao de nova coluna com um novo Nivel...
dataset$Level3 = dataset$Level^3 #criacao de nova coluna com um novo Nivel...
dataset$Level4 = dataset$Level^4 #criacao de nova coluna com um novo Nivel...
poly_reg = lm(formula =Salary  ~., data = dataset)

# Visualising the Linear Regression Results
#install.packages('ggplot2')
library(ggplot2)
ggplot()+
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level , y = predict(lin_reg, newdata = dataset)),
            color = 'blue')+
  ggtitle('Truth or Bluff(Linear regression)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Polynomial Regression Results
ggplot()+
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level , y = predict(poly_reg, newdata = dataset)),
            color = 'blue')+
  ggtitle('Truth or Bluff(Polynomial regression)') +
  xlab('Level') +
  ylab('Salary')

# Predicting a new result with Linear Regression
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# Predicting a new result with Polynomial Regression
y_pred = predict(poly_reg, data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4))