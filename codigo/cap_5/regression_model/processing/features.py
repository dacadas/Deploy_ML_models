from typing import List

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

#####################################################################
# Transformer para calcular a diferença entre variáveis temporais
class TemporalvariableTransformer( BaseEstimator , TransformerMixin):
    
    # Inicializa classe
    def __init__(self, variables, reference_variable):
        if not isinstance(variables,  list):
            raise ValueError('variables devem ser uma lista')
            
        # Armazena as variáveis e a variável de referência
        self.variables = variables
        self.reference_variable = reference_variable
    
    # crear metodo - Necessário para compatibilidade com sklearn Pipeline
    def fit(self, X, y=None):
        return self
    
    # Método transform custom -- Calcula a diferença entre a variável de referência e cada variável temporal
    def transform(self,X):
        # Cria uma cópia para não modificar o DataFrame original
        X = X.copy()
        
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]
        
        return X

#####################################################################
# Transformer para recategorizar variáveis categóricas com um mapeamento predefinido
class Mapper( BaseEstimator , TransformerMixin):   
    
    # Inicializa  classe
    def __init__(self, variables, mappings):
        if not isinstance(variables,  list):
            raise ValueError('variables devem ser uma lista')
                
        # Armazena as variáveis e o dicionário de mapeamento
        self.variables = variables
        self.mappings = mappings   
        
    # crear metodo - Necessário para compatibilidade com sklearn Pipeline
    def fit(self, X, y=None):
        return self
    
    # Método transform custom -- Aplica o mapeamento às variáveis especificadas
    def transform(self,X):
        # Cria uma cópia para não modificar o DataFrame original
        X = X.copy()
        
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)        
        return X
#####################################################################
# coverte variaveis a tipo objeto
class CastVariablesAsObject(BaseEstimator, TransformerMixin):
    """
    Converte variáveis para o tipo 'object'.

    Útil quando algum transformer reconstrói o DataFrame e o pandas
    volta a inferir tipos numéricos.
    """

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for var in self.variables:
            X[var] = X[var].astype("object")

        return X    