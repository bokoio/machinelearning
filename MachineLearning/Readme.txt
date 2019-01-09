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






