Instalado o Anaconda 
https://repo.continuum.io/archive/
versao :Anaconda3-2018.12-Linux-x86_64.sh

export PATH="~/anaconda3/bin/"

No terminal = anaconda-navigator
OU 
Criar inone no desktop:
[Desktop Entry]
Version=1.0
Type=Application
Name=Anaconda-Navigator
GenericName=Anaconda
Comment=Scientific PYthon Development EnviRonment - Python3
Exec=bash -c 'export PATH="/home/pippo/anaconda3/bin:$PATH" && /home/pippo/anaconda3/bin/anaconda-navigator'
Categories=Development;Science;IDE;Qt;Education;
Icon=spyder3
Terminal=false
StartupNotify=true
MimeType=text/x-python;

Para instalar o R:
https://cran.r-project.org
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/
sudo apt-get update
sudo apt-get install r-base

RStudio IDE para desenvolver em R
www.rstudio.com

-----------
S1 Toda para a instalaçao do ambiente
-----------
S2A11 - Get the data set:

Baixado o primeiro dataset e a estrutura de pastas que serao utilizadas no decorrer do curso.
https://www.superdatascience.com/machine-learning/
PART 1. DATA PREPROCESSING
Data_Preprocessing.zip

Dentro do arquivo Data.csv
Esistem 4 colunas A B C D
Country,Age,Salary,Purchased
As 3 primeiras colunas (Country,Age,Salary) sao variaveis independentes
e a ultima coluna (Purchased) é uma variavel dependente
Nos modelos de machinelearning iremos utilizar variaveis independentes para prever o valor das variaveis dependentes


S2A12 - Importing the Libraries:
Criei o arquivo:
data_preprocessing_template.py
Onde foram importadas as librerias :
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Criado tambem o arquivo
data_preprocessing_template.r
onde nao foram importadas librerias porque o R ja carrega as librerias necessarias.

S2A13 - Importing the Dataset:

Importando o dataset no python
1-Indicar a fonte do dado onde esta localizado o arquivo .csv
No caso do exemplo:
/home/pippo/Dev/MachineLearning/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/S2/A2/Data.csv

que é o valor da working directory:
wdir='/home/pippo/Dev/MachineLearning/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/S2/A2'

Apos o "marcar" o working directory e o import do dataset:
dataset = pd.read_csv('Data.csv')
Foi definida a matriz de recursos e o vetor das variaveis dependentes
(a matriz contem as 3 colunas com as variaveis independentes)
X = dataset.iloc[:, :-1].values
iloc[:, :-1]
o primeiro : é referente as linhas (quer dizer que iremos carregar todas as linhas)
o segundo : é referente as colunas iremos pegar todas menos a ultima coluna.

O vetor de variaveis dependentes foi criado da seguinte maneira:
y = dataset.iloc[:, 3].values
Lendo o dataset
pegando todas as linhas
pegando somente a ultima coluna
iloc[:, 3]
o numero 3 é referente ao index da coluna no dataset, que no caso do python o index inicia no zero 0


Em R:

No RStudio:
Painel inferior esquerdo
Files>PastaOndeEstaOData.csv
Botao>More> Set as working Directory.

A console retornara uma mensagem parecida com essa:
setwd("~/Dev/MachineLearning/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/S2/A2")

Codigo que carrega o dataset:
dataser = read.csv('Data.csv')
Em R nao precisa criar a matriz e nem o vetor com os dados

S2A14 - For Python learners, summary of Object-oriented programming: classes & objects:

PS: IMPORTANT NOTE. In the Python part of the next tutorial, ‘NaN’ has to be replaced by ‘np.nan’. This is due to an update in the sklearn library.


S2A15 - Missing Data:

Preparar os dados para que o machineLearning model funcione corretamente:
Tratamento para quando um campo nao esta preenchido.
Nao remover as linas do dataset
Para preencher os valores em branco, sera feita a media dos valores da coluna em que o dado estiver faltando:
no caso temos um campo vazio na coluna B (Age) e na coluna C (Salary)

Em Python:
Import do modulo sklearn
que é um modulo onde existem varios modelos de funçoes para trabalhar com dados e machineLearning.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

Em R:

Para cada um dos campos do dataset onde possivelmente pode haver um campo vaziu
é utilizada uma funçao ja nativa do R ifelse:
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Age(dataset com a coluna Age)
ifelse(is.na(dataset$Age), // Primeiro parametro verifica se o campo esta vazio
ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), //funçao nativa do R para fazer a media (average) caso o campo esteja vazio
dataset$Age) // terceiro parametro é o que fazer em caso do campo estar populado, somente manter o valor.


S2A16 - Categorical Data:

No DataSet tem duas variaveis de categorias (Categorical Variables)
Sao elas as colunas A e D (Country e Purchased)
Os modelos trabalham com equaçoes, portanto devemos transformar as strings em numeros.

Em Python

O codigo abaixo, agrupa os valores iguais por numero:
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

Franca = 0
Spain = 2
Germany = 1

Mas esse modelo tem um problema porque como fazer para dizer que a Franca é maior que a Spain e que a Germany é Maior que a Spain.
Pra isso serao criados 3 novas colunas que irao fazer referencia ao valor
Franca | Spain | Germany
  1    |   0   |     0
  0    |   1   |     0
  0    |   0   |     1
  .....
  .....

para fazer essa alteraçao importei a classe OneHotEncoder
o primeiro parametro nao é muito relevante mas o segundo parametro categorical_features sim é.
usamos esse parametro: 
   - array of indices: Array of categorical feature indices.
A funçao ficou dessa forma:
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
Da forma como ficou o DataSet o campo Country nao fica qualificado, mantendo os dados imparciais(ver imagem
/home/pippo/Dev/MachineLearning/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/S2/A2/S2A16.png)

Essa forma so foi utilizada para nao qualificar os dados do campo Country, ja o campo Purchased que so possui 2 valores Yes ou No pode ser alterado utilizando somente essa funçao:
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

O no ficou com valor 0 e o yes com valor 1

Em R:
é muito mais simples encoded os valores:
A açao é direta no campo :
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))



S2A18 - Spliting the Datase into The Training set and Test set:
To avoid this, you just need to replace:
from sklearn.cross_validation import train_test_split 
by
from sklearn.model_selection import train_test_split

Em todos os modelos de dados do machineLearning os dados devem sempre serem divididos em outros 2 datasets:
Trainig and Test

Em Python:
o commando abaixo cria os novos datasets para x e y
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
o parametro test_size indica o tamanho do modelo em percentual fronte ao original.
No exemplo estamos usando 20% de 100% do original.
O random_state é somente para ter os resultados dos mostrados no curso.

Em R:

Para fazer o split em R é necessario instalar uma biblioteca a caTools:
para instalar uma biblioteca em R:
install.packages('')//nome da biblioteca.
apos excutar no terminal comentar a linha
para usar essa nova biblioteca
library(caTools)



S2A19 - Feature Scaling:

As colunas B e C (Age e Salary) estao em escalas diferentes, para que uma variavel nao :
Para normalizar esses valores (colocando eles no mesmo rang e na mesma escala, para que nenhuma variavel seja dominada por outra)
Podem ser utilizadas formulas matematicas, as 2 mais comuns sao:
Standardisation e Normalisation

Standardisation:
Xstand = X - mean(X)
        _________________
         standard deviation(X)


Normalisation:

Xnorm = X - mean(X)
        _________________
        max(x)-min(x)

Em Python:
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #primeiro transformar depois fazer o fit
X_test = sc_X.transform(X_test)#aqui nao precisa fazer o fit somente transformar

Em R:
training_set = scale(training_set)
test_set = scale(test_set)
Ao executar os comandos acima, retornou o erro:
Error in colMeans(x, na.rm = TRUE) : 'x' must be numeric
Isso ocorre porque mesmo as colunas possuindo numeros, o campo ainda é um char, devido a alteraçao feita anteriormente no Categorical Data.
Para corrigir isso limitamos a scala para somente 2 colunas (B e C)

training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])



S2A21 - Data Preprocessing Template

Refactoring do template criado em R e Python


S4A22 - How to get the DataSet
https://www.superdatascience.com/machine-learning/
Section 4. Simple Linear Regression
Simple_Linear_Regression.zip

S4A23 - Dataset and Business problem Description:
dataset tem somente 2 colunas A e B (YearsExperience, Salary)

Qual è a co-relaçao entre os anos de experiencia e o salario?

S4A24 - Simple Linear Regression Intuition Step1:
Formula para o simple linear regression:
y = B0 + B1*X1

Y = dependent variable(dv) é o que esta se tentando explicar(como os salarios mudam com os anos de experiencia no trabalho.)
X = Independet variable(iv) 
B1 = Coeficiente para a variavel independente X1
B0 = Constante

y = B0 + B1*X1
Salary = b0 + b1 * Experience
B0 = 30k (pessoa sem nenhuma experiencia)


S4A25 - Simple Linear Regression Intuition Step2:

para chegar no B0 que é a linha inicial a formula é a seguinte:
sum (y - y^)² -> min
y -salario mais alto
y^ -diferença entre o salario mais alto e a linha.


S4A26 - Simple Linear Regression in Python Step1:


Fizemos somente a preparaçao dos dados como fizemos no modelo anterior com o preprocessing_template.

(limpar as variaveis toda a vez que trocar o DB de lugar.)

S4A27 - Simple Linear Regression in Python Step2:
Fitting Simple Linear Regrssion to the Training set

#Fitting Simple Linear Regrssion to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


S4A28 - Simple Linear Regression in Python Step3:

#Predicting the Test set results

y_pred //vector que ira prever as dependentes variaveis


S4A29 - Simple Linear Regression in Python Step4:
#Visualising the Training set results
OS resultados estao nas imagens, e esse é um modelo linear de machineLearning.
que prediz os salarios dos empregados.


S4A30 - Simple Linear Regression in R Step1:

S4A31 - Simple Linear Regression in R Step2:
Aqui é onde estamos treinando o nosso modelo(Taining Test)
~ o til representa proporçao.
no console do R digitando:
summary(regressor) //variavel com os dados da regreçao
nos retorna um sumario dos dados elaborados por:
lm(formula = Salary ~ YearsExperience,
                data = training_set)

é importanto verificar quantas estrelas a variavel recebeu.

S4A32 - Simple Linear Regression in R Step3:
#Predicting the Test set results
Previsao de resultados


S4A33 - Simple Linear Regression in R Step4:
para gerar os graficos em R foi necessario importar uma biblioteca:
ggplot2
install.packages('ggplot2')



S5A35 - Dataset and Business problem Description For Multiple Linear Regression
DataSet: 50_Startups.csv
O dataset possui somente 5 colunas:
R&D Spend,	Administration,	Marketing Spend,	State,	Profit



Criar um modelo para prever em quais empresas investir com os dados que temos:
Necessidade de saber se existe alguma corelaçao entre o lucro e o investimento em R&D ou em Admin ou em Marketing e em qual estado as empresas estao tendo melhores lucros.



S5A36 - Multiple Linear Regression intuition Step1:

A foruma é parecida porem dessa vez sao adicionados mais pares de comparaçao de acordo com a quantidade de variaveis independentes que estao no modelo:
y = B0 + B1*X1 + B2*X2 + B3*X3 + ....+Bn*Xn

S5A37 - Multiple Linear Regression intuition Step2:

Assumptions(pressupostos) of a Linear Regression:
	1) Linearity
	2) Homoscedasticity
	3) Multivariae normality
	4) Independence of errors
	5) Lack of multicollinerity

Para construir um bom modelo linear é preciso que essas 5 respostas sejam verdadeiras.


S5A38 - Multiple Linear Regression intuition Step3:

Dumy variables:
Dependente = Profit
Independentes = R&D Spend,	Administration,	Marketing Spend,	State

Criar um Linear Regression

Profit,  R&D Spend,	Administration,	Marketing Spend,	State(categorical data)
y=b0+     B1*X1   +   B2*X2        + B3*X3          + ????

Dammy variables funcionam como um boolean para identificar um dado de categoria
é o mesmo que ja foi feito antes:
New York | California 
  1      |   0   
  0      |   1   
  0      |   1   
  1      |   0   
  .....
  .....
y=b0+     B1*X1   +   B2*X2        + B3*X3          + B4*D1 (d for dummy)


S5A38 - Multiple Linear Regression intuition Step4:
Pegadinha das variaveis dummy:
Inserindo todas as variaveis dummy no mesmo modelo, nao sera possivel distinguir os efeitos de D1 sobre D2 nao funcionando corretamente.
Para isso sempre omitir uma das variaveis dummy, por exemplo:
no dataset dessa aula esistem 3 estados:
New York
California
Florida
cada um gera uma dummy, no modelo devemos inserir somente 2 variaveis no modelo.


S5A40 - Prerequisites: What is the P-Value:
https://www.wikihow.com/Calculate-P-Value:
P value is a statistical measure that helps scientists determine whether or not their hypotheses are correct. P values are used to determine whether the results of their experiment are within the normal range of values for the events being observed. Usually, if the P value of a data set is below a certain pre-determined amount (like, for instance, 0.05), scientists will reject the "null hypothesis" of their experiment - in other words, they'll rule out the hypothesis that the variables of their experiment had no meaningful effect on the results. Today, p values are usually found on a reference table by first calculating a chi square value.

https://www.mathbootcamps.com/what-is-a-p-value/
https://www.youtube.com/watch?v=eyknGvncKLw
In statistics, we always seem to come across this p-value thing. If you have been studying for a while, you are used to the idea that a small p-value makes you reject the null hypothesis. But what if I asked you to explain exactly what that number really represented!?

Understanding the p-value will really help you deepen your understanding of hypothesis testing in general. Before I talk about what the p-value is, let’s talk about what it isn’t.

The p-value is NOT the probability the claim is true. Of course, this would be an amazing thing to know! Think of it “there is 10% chance that this medicine works”. Unfortunately, this just isnt the case. Actually determining this probability would be really tough if not impossible!
The p-value is NOT the probability the null hypothesis is true. Another one that seems so logical it has to be right! This one is much closer to the reality, but again it is way too strong of a statement.
The p-value is actually the probability of getting a sample like ours, or more extreme than ours IF the null hypothesis is true. So, we assume the null hypothesis is true and then determine how “strange” our sample really is. If it is not that strange (a large p-value) then we don’t change our mind about the null hypothesis. As the p-value gets smaller, we start wondering if the null really is true and well maybe we should change our minds (and reject the null hypothesis).

A little more detail: A small p-value indicates that by pure luck alone, it would be unlikely to get a sample like the one we have if the null hypothesis is true. If this is small enough we start thinking that maybe we aren’t super lucky and instead our assumption about the null being true is wrong. Thats why we reject with a small p-value.

A large p-value indicates that it would be pretty normal to get a sample like ours if the null hypothesis is true. So you can see, there is no reason here to change our minds like we did with a small p-value.

S5A41 - Multiple Linear Regression intuition Step5:

Building A Model(Step by Step):


Selecionar bem as colunas candidatas a serem variaveis Dependentes:
Mas porque nao usar todas as variaveis:

1) Grabage in Garbage Out
2) Se torna um problema quando temos uma grande quantidade de variaveis para explicar(a matematica e os resultados por traz dessas previsoes feitas por essa variaveis)

5 Metodos de construçao de modelos:
PDF explicando esta em /home/pippo/Dev/MachineLearning/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/Step-by-step-Blueprints-For-Building-Models.pdf


1) All-in
2) Backward Elimination      -|
3) Forward Selectio           |- Stepwise Regression 
4) Bidirectional Elimination -|
5) Score Comparsion


O modelo que iremos construir é do tipo Backward Elimination


S5A43 - Multiple Linear Regression intuition In Python Step2:

DataPrecessing;
...
#Avoiding the dummy variable Trap
#Nao é necessario fazer manualmente a libreria faz isso por nos mas para exemplo vale:
X = X[:, 1:]
...

S5A44 - Multiple Linear Regression intuition In Python Step3:

# Fitting Multiple Linear Regression  to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#Predicting the Test set results
y_pred = regressor.predict(X_test)



S5A45 - Multiple Linear Regression intuition In Python Backward Elimination Prepare:

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm 


S5A46 - Multiple Linear Regression intuition In Python Backward Elimination HomeWork:
#Backwad Elimination

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr =np.ones((50, 1)).astype(int) , values =  X ,axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y ,exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y ,exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y ,exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y ,exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y ,exog = X_opt).fit()
regressor_OLS.summary()



S5A49 - Multiple Linear Regression In R Step1:
Trasnformar texto em categorical data

dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))


S5A50 - Multiple Linear Regression In R Step2:


A formula para o multi linear regression:

Profit é igual a combinaçao linear das variaveis independentes (no caso sao as 4 primeiras colunas do banco)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State)
Forma sintetizada de escrever a mesma formula
regressor = lm(formula = Profit ~ .)

no terminal summary(regressor)
retornam dados, uma parte muito importante é o trexo abaixo:
Coefficients:
                  Estimate Std. Error t value Pr(>|t|)    
(Intercept)      4.965e+04  7.637e+03   6.501 1.94e-07 ***
R.D.Spend        7.986e-01  5.604e-02  14.251 6.70e-16 ***
Administration  -2.942e-02  5.828e-02  -0.505    0.617    
Marketing.Spend  3.268e-02  2.127e-02   1.537    0.134    
State2           1.213e+02  3.751e+03   0.032    0.974    
State3           2.376e+02  4.127e+03   0.058    0.954    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


Indica os valores recebidos apos a aplicaçao da formula
quanto menor melhor.
No caso temos um valor com 3 estrelas esse é o valor indicado para ser usado.

Isso quer dizer que o melhor tipo de investimento é em empresas que tenham bastante investimento em R & D
pois esse tipo de investimento esta ligado a maior lucratividade.

Olhando para esses resultados a formula poderia ser escrita dessa forma:
regressor = lm(formula = Profit ~ R.D.Spend, 
              data = training_set)
Se tornado uma linear regression

S5A51 - Multiple Linear Regression In R Step3:

#Predicting the Test set results
Predict results:

y_pred = predict(regressor,newdata = test_set)


S5A52 - Multiple Linear Regression In R Backward Elimination HomeWork:


#Building the optimal model using Backward Elimination

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend ,
               data = dataset)
summary(regressor)

if you are also interested in an automatic implementation of Backward Elimination in R, here it is:

backwardElimination <- function(x, sl) {
    numVars = length(x)
    for (i in c(1:numVars)){
      regressor = lm(formula = Profit ~ ., data = x)
      maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
      if (maxVar > sl){
        j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
        x = x[, -j]
      }
      numVars = numVars - 1
    }
    return(summary(regressor))
  }
  
  SL = 0.05
  dataset = dataset[, c(1,2,3,4,5)]
  backwardElimination(training_set, SL)


S6A55 - Polynomial linear Regression Intuition

Formula:
y = b0 +b1x1 + b2x1² +....+bnx1n

quando se usa polinomial regression:
normalmente utilizada para demonstrar o crecimento de doenças ou populaçao....


S6A57 - Polynomial Regression in Python Step1:
Preparaçao do dataset
Nao foi realizado split do data set pois o objectivo aqui é intuir se o salario que o empregado disse que estava ganhando na outra empresa é verdadeiro ou nao, a empresa anterior enviou os possiveis salaraios com base no cargo.


S6A58 - Polynomial Regression in Python Step2:

Aqui iremos criar 2 modelos:
1) linear regression
2) Polynomial regression

Porque desses 2 modelos? Para poder comparar o resultado dos 2.

# Fitting Simple Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the Training set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


S6A59 - Polynomial Regression in Python Step3:

Graphics with results(S6A59.odc)



S6A60 - Polynomial Regression in Python Step4:

# Predicting a new result with Linear Regression
for_lin_pred = np.array(6.5).reshape(1,-1)
lin_reg.predict(for_lin_pred)

# Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(for_lin_pred))



S6A61 - Python Regression Template:
Criaçao do template para usar nos modelos de Regreçao.

S6A63 - Polynomial Regression in R Step1:

S6A65 - Polynomial Regression in R Step4:
# Predicting a new result with Linear Regression
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# Predicting a new result with Polynomial Regression
y_pred = predict(poly_reg, data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4))


S6A66 - R Regression Template:


S7A68 - 

