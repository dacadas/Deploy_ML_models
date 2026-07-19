import numpy as np
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
    # tol: frequência mínima para que uma categoria NAO seja considerada rara
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
    
#####################################################################
# Transformer para codificar variáveis categóricas em valores numéricos   
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    # Inicializa a classe
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            raise ValueError('variables deve ser uma lista')
        
        # Armazena as variáveis
        self.variables = variables
        
    # Cria um mapeamento ordinal baseado na média da variável alvo
    def fit(self, X, y):
        # Combina os dados de entrada com a variável alvo
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        # cria um diccionario para armazenar mapeamentos
        self.encoder_dict_ = {}

        for var in self.variables:
            # Ordena as categorias pela média da variável alvo
            t = temp.groupby([var])["target"].mean().sort_values(ascending=True).index
            
            # Atribui um código numérico para cada categoria
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # Cria uma cópia para não modificar o DataFrame original
        X = X.copy()
            
        # mapeia 
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

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