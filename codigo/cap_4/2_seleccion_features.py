import pandas as pd
import numpy as np
import os
import random


# para construir os modelos
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# para visualizar todas as colunas no dataframe
pd.pandas.set_option('display.max_columns', None)


def criar_pasta(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    
# Para fins de reprodutibilidade
semilla  = 42
np.random.seed( semilla )
random.seed(semilla) 


#### Carregar o dataset
path_data = 'D:/Diego/Curso Udemy/Deploy_ML_models/dataset/dado_preprocesado/'

X_train = pd.read_csv(path_data + 'xtrain.csv')
X_test  = pd.read_csv(path_data + 'xtest.csv')
y_train = pd.read_csv(path_data + 'ytrain.csv')
y_test  = pd.read_csv(path_data + 'ytest.csv')

####
# Faremos o ajuste do modelo e a seleção de características em apenas
# algumas linhas de código

# Em primeiro lugar, especificamos o modelo de Regressão Lasso e selecionamos um
# alfa adequado (equivalente à penalização). Quanto maior o alfa, 
# menos características serão selecionadas.

# Em seguida, usamos o objeto SelectFromModel do sklearn, que selecionará 
# automaticamente as características cujos coeficientes não são zero


sel_ = SelectFromModel(Lasso(alpha=0.001, random_state=semilla))

# treinar o modelo Lasso e selecionar características
sel_.fit(X_train, y_train)

sel_.get_support().sum()

# Vamos visualizar as características que foram selecionadas. 
# (características selecionadas marcadas como True)
sel_.get_support()

# vamos imprimir o número de características totais e selecionadas

# é assim que podemos criar uma lista das características selecionadas
selected_feats = X_train.columns[(sel_.get_support())]

# vamos imprimir algumas estatísticas
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feats)))
print('features with coefficients shrank to zero: {}'.format(
    np.sum(sel_.estimator_.coef_ == 0)))



# imprimir as características selecionadas
selected_feats

pd.Series(selected_feats).to_csv(path_data + 'selected_features.csv', index=False)