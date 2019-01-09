#Simple Linear Regression
dataset = read.csv('Salary_Data.csv')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3) 
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting Simple Linear Regrssion to the Training set
regressor =  lm(formula = Salary ~ YearsExperience,
                data = training_set)
#Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

#Visualising the Training set results
#install.packages('ggplot2')
library(ggplot2)
ggplot()+
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Traning set)') +
  xlab('Years of Experience') +
  ylab('Salary')

#install.packages('ggplot2')
library(ggplot2)
ggplot()+
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of Experience') +
  ylab('Salary')










