# to handle datasets
import pandas as pd
import numpy as np
import os
import random
# for plotting
import matplotlib.pyplot as plt

# for the yeo-johnson transformation
import scipy.stats as stats

# to save the model
import joblib

# para visualizar todas as colunas no dataframe
pd.pandas.set_option('display.max_columns', None)


def criar_pasta(path):
    if not os.path.exists(path):
        os.makedirs(path)
       
# Para fins de reprodutibilidade
semilla  = 42
np.random.seed( semilla )
random.seed(semilla) 


path_projecto = 'D:/Diego/Curso Udemy/Deploy_ML_models/dataset/'
path_data = path_projecto + 'dado_preprocesado/'
path_model = path_projecto + '/modelos/'


#### Carregar o dataset
data = pd.read_csv(path_projecto + 'test.csv')
data.drop('Id', axis=1, inplace=True)

## carregar modelo
lin_model = joblib.load(path_model + 'linear_regression.joblib')
lin_model_sel = joblib.load(path_model + 'linear_regression_sel.joblib')

scaler = joblib.load(path_data + 'minmax_scaler.joblib')




## transformar dados para modelo

# es muy costos computacionalmente, hay que guardar un monton de parametros manualmente
# en relacion al entrenamiento


