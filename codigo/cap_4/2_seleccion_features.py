import pandas as pd
import numpy as np
import os
import random

# for plotting
import matplotlib.pyplot as plt

# to build the models
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


def criar_pasta(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    
# Para fins de reprodutibilidade
semilla  = 13
np.random.seed( semilla )
random.seed(semilla) 


#### Carregar o dataset
path_data = 'D:/Diego/Curso Udemy/Deploy_ML_models/dataset/dado_preprocesado/'


X_train = pd.read_csv(path_data + 'xtrain.csv')
X_test = pd.read_csv(path_data + 'xtest.csv')