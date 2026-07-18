import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

#####################################################################
# Transformer para calcular a diferença entre variáveis temporais
class TemporalvariableTransformer( BaseEstimator , TransformerMixin):
    
    # Inicializar clase
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
    
    # Inicializar clase
    def __init__(self, variables, mappings):
        if not isinstance(variables,  list):
            raise ValueError('variables devem ser uma lista')
                
        # Armazena as variáveis e o dicionário de mapeamento
        self.variables = variables
        self.mappings = mappings   
        
    # crear metodo - Necessário para compatibilidade com sklearn Pipeline
    def fit(self, X, y=None):
        return self
    
    # Método transform custom -- asigna valor a variable --mapeada
    def transform(self,X):
        # Cria uma cópia para não modificar o DataFrame original
        X = X.copy()
        
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)        
        return X
    
#####################################################################
# Transformer para imputar valores numéricos ausentes com a média
class MeanImputer(BaseEstimator, TransformerMixin):

    # Inicializa a classe
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            raise ValueError("variables deve ser uma lista")

        # Armazena as variáveis
        self.variables = variables

    # Calcula e armazena a média de cada variável
    def fit(self, X, y=None):
        # Armazena a média de cada variável em um dicionário
        self.imputer_dic_ = X[self.variables].mean().to_dict()
        return self

    # Substitui os valores ausentes pela média correspondente
    def transform(self, X):
        # Cria uma cópia para não modificar o DataFrame original
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].fillna(self.imputer_dic_[feature])

        return X


#####################################################################
# Transformer para agrupar categorias raras em uma única categoria ("Rare")
class RarelabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    # Inicializa a classe
    # tol: frequência mínima para que uma categoria não seja considerada rara
    def __init__(self, tol=0.05, variables=None):
        if not isinstance(variables, list):
            raise ValueError("variables deve ser uma lista")

        # Armazena o limite de frequência e as variáveis
        self.tol = tol
        self.variables = variables

    # Identifica e armazena as categorias frequentes
    def fit(self, X, y=None):
        # Dicionário que armazenará as categorias não raras
        self.encoder_dic_ = {}

        for var in self.variables:
            # Calcula a frequência relativa de cada categoria
            t = X[var].value_counts(normalize=True)

            # Armazena as categorias cuja frequência é maior ou igual ao limite
            self.encoder_dic_[var] = list(t[t >= self.tol].index)

        return self

    # Substitui as categorias raras pela categoria "Rare"
    def transform(self, X):
        # Cria uma cópia para não modificar o DataFrame original
        X = X.copy()

        for feature in self.variables:
            X[feature] = np.where(  X[feature].isin(self.encoder_dic_[feature]),
                                    X[feature],
                                    "Rare"  )

        return X