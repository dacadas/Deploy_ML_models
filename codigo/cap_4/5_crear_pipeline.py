# data manipulation and plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
# for saving the pipeline
import joblib

# from Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer  # para skewd
from sklearn.preprocessing import MinMaxScaler

# from feature-engine
from feature_engine.imputation import (
    AddMissingIndicator,  # crea columna para indicar cuando falta algun valor
    MeanMedianImputer,    # imputar media o mediana
    CategoricalImputer,   # imputar mas frecuente ou colocar 'missing'
)

from feature_engine.encoding import (
    RareLabelEncoder,     # remover labels raros
    OrdinalEncoder,       # genera clases dentro de variable para ser mas entendible
)

from feature_engine.transformation import (
    LogTransformer,          # transformacion de distribucion
    YeoJohnsonTransformer,
)

from feature_engine.selection import DropFeatures
from feature_engine.wrappers import SklearnTransformerWrapper

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# Para fins de reprodutibilidade
semilla  = 13
np.random.seed( semilla )
random.seed(semilla) 


#### Carregar o dataset
path_data = 'D:/Diego/Curso Udemy/Deploy_ML_models/dataset/'

data = pd.read_csv(path_data + 'train.csv')
print(data.shape)


## Elimina a coluna ID que não traz informação
# data.drop('Id', axis=1, inplace=True)

###############################################################################
## Separa em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split( data.drop(['SalePrice'], axis=1), # Variáveis preditivas (X)
                                                     data['SalePrice'], # Alvo (y)
                                                     test_size    = 0.1, # Proporção do dataset a ser alocada para o conjunto de teste
                                                     random_state = semilla, # Estamos definindo a semente (seed) aqui
                                                    )

## Transformar o alvo para logaritmo para deixá-lo mais gaussiano
y_train = np.log(y_train)
y_test = np.log(y_test)


#############################################################
#### Engenharia de dados sobre
### a- Valores faltantes
## a1 -- Categóricos 
# Primeiro identificar quais variáveis são categóricas
cat_vars = [var for var in data.columns if data[var].dtype == 'O']

# Para este dataset específico, adicionar MSSubClass à lista de variáveis categóricas
cat_vars = cat_vars + ['MSSubClass']

# Converter todas as variáveis para categóricas
X_train[cat_vars] = X_train[cat_vars].astype('O')
X_test[cat_vars]  = X_test[cat_vars].astype('O')

# Separar todas as variáveis categóricas com perda de dados (conjunto de treino)
cat_vars_with_na = [ var for var in cat_vars if X_train[var].isnull().sum() > 0]

# Exibe a porcentagem de valores faltantes por variável
X_train[ cat_vars_with_na ].isnull().mean().sort_values(ascending=False)

# Variáveis para imputar com a string 'Missing' (10% é um valor arbitrário)
with_string_missing = [ var for var in cat_vars_with_na if X_train[var].isnull().mean() > 0.1]

# Variáveis para imputar com a categoria mais frequente
with_frequent_category = [ var for var in cat_vars_with_na if X_train[var].isnull().mean() < 0.1]



############# Parte del pipeline
### a- Valores faltantes
## a1 -- Categóricos 
## remeplace valores ausentes missing con nuevo rotulo: "Missing"
# configurar la clase
cat_imputer_missing = CategoricalImputer( imputation_method='missing', variables = with_string_missing)
#  ajustar la clase al conjunto de entrenamiento
cat_imputer_missing.fit(X_train)
# La clase aprende y almacena los parametros
cat_imputer_missing.imputer_dict_

## reemplace NA por missing
# podemos alamcenar esta clase con joblib
X_train = cat_imputer_missing.transform(X_train)
X_test  = cat_imputer_missing.transform(X_test)

## reemplace valores ausentes missing con categoria que mas se repite
# configurar la clase
cat_imputer_frequent = CategoricalImputer( imputation_method='frequent', variables=with_frequent_category)
#  ajustar la clase al conjunto de entrenamiento
cat_imputer_frequent.fit(X_train)
# La clase aprende y almacena los parametros
cat_imputer_frequent.imputer_dict_

## reemplace NA por missing
# podemos alamcenar esta clase con joblib
X_train = cat_imputer_frequent.transform(X_train)
X_test  = cat_imputer_frequent.transform(X_test)

# # check that we have no missing information in the engineered variables
# X_train[cat_vars_with_na].isnull().sum()
# # check that test set does not contain null values in the engineered variables
# [var for var in cat_vars_with_na if X_test[var].isnull().sum() > 0]

## a2 -- Numéricas 
# Primeiro identificar quais variáveis são numéricas
num_vars = [    var for var in X_train.columns if var not in cat_vars and var != 'SalePrice' ]

# Verificar dados faltantes
vars_with_na = [    var for var in num_vars    if X_train[var].isnull().sum() > 0]

# # Exibe a porcentagem de valores faltantes por variável
# X_train[vars_with_na].isnull().mean().sort_values(ascending=False)

# Substituir valores faltantes pela média no treino e teste e criar uma nova variável binária para indicar onde houve a mudança
missing_ind = AddMissingIndicator(variables=vars_with_na)
missing_ind.fit(X_train)

X_train = missing_ind.transform(X_train)
X_test  = missing_ind.transform(X_test)

# # check the binary missing indicator variables
# X_train[['LotFrontage_na', 'MasVnrArea_na', 'GarageYrBlt_na']].head()

## reemplace valores faltantes con la media
# configurar la clase
mean_imputer = MeanMedianImputer(imputation_method='mean', variables=vars_with_na)
#  ajustar la clase al conjunto de entrenamiento
mean_imputer.fit(X_train)
# La clase aprende y almacena los parametros
mean_imputer.imputer_dict_

# aplicar
X_train = mean_imputer.transform(X_train)
X_test  = mean_imputer.transform(X_test)

# Verifica se não temos mais valores faltantes nas variáveis modificadas
# X_train[vars_with_na].isnull().sum()

# check that test set does not contain null values in the engineered variables
# [var for var in vars_with_na if X_test[var].isnull().sum() > 0]

##################
### b- Variáveis temporais
# Para este dataset específico, será feita uma relação entre variáveis de tempo
def elapsed_years(df, var):
    # Captura a diferença entre a variável de ano e o ano em que a casa foi vendida
    df[var] = df['YrSold'] - df[var]
    return df
for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    X_train = elapsed_years(X_train, var)
    X_test  = elapsed_years(X_test, var)


# now we drop YrSold
drop_features = DropFeatures(features_to_drop=['YrSold'])
X_train = drop_features.fit_transform(X_train)
X_test = drop_features.transform(X_test)

##################
### c- Variáveis com distribuição não gaussiana (aplicar transformação,
# seja log, seja Yeo-Johnson para deixá-las mais gaussianas)
# Especificamente aqui, na análise exploratória vimos que o log ajuda
log_transformer = LogTransformer(variables=["LotFrontage", "1stFlrSF", "GrLivArea"])

X_train = log_transformer.fit_transform(X_train)
X_test  = log_transformer.transform(X_test)

# check that test set does not contain null values in the engineered variables
# [var for var in ["LotFrontage", "1stFlrSF", "GrLivArea"] if X_test[var].isnull().sum() > 0]

# A transformação Yeo-Johnson descobre o melhor expoente para transformar a variável
# Ela precisa aprender isso a partir do conjunto de treino: 
yeo_transformer = YeoJohnsonTransformer(    variables=['LotArea'])

X_train = yeo_transformer.fit_transform(X_train)
X_test = yeo_transformer.transform(X_test)

# La clase aprende y almacena los parametros
yeo_transformer.lambda_dict_


# Havia algumas variáveis muito assimétricas, vamos transformá-las em variáveis binárias.
skewed = [
    'BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch',
    '3SsnPorch', 'ScreenPorch', 'MiscVal'
]
binarizer = SklearnTransformerWrapper( transformer=Binarizer(threshold=0), variables=skewed )

X_train = binarizer.fit_transform(X_train)
X_test = binarizer.transform(X_test)

### Categorical variables -- aplicar mapping
### Codificar por qualidade as variáveis categórias (remapear)
qual_mappings = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'Missing': 0, 'NA': 0}
qual_vars = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
             'HeatingQC', 'KitchenQual', 'FireplaceQu',
             'GarageQual', 'GarageCond',            ]
for var in qual_vars:
    X_train[var] = X_train[var].map(qual_mappings)
    X_test[var] = X_test[var].map(qual_mappings)


exposure_mappings = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
var = 'BsmtExposure'
X_train[var] = X_train[var].map(exposure_mappings)
X_test[var] = X_test[var].map(exposure_mappings)

finish_mappings = {'Missing': 0, 'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
finish_vars = ['BsmtFinType1', 'BsmtFinType2']
for var in finish_vars:
    X_train[var] = X_train[var].map(finish_mappings)
    X_test[var] = X_test[var].map(finish_mappings)
    
garage_mappings = {'Missing': 0, 'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
var = 'GarageFinish'
X_train[var] = X_train[var].map(garage_mappings)
X_test[var] = X_test[var].map(garage_mappings)  

fence_mappings = {'Missing': 0, 'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
var = 'Fence'
X_train[var] = X_train[var].map(fence_mappings)
X_test[var] = X_test[var].map(fence_mappings)  


##################
### d- Variáveis categóricas: eliminar etiquetas raras (Rare labels)
# Para as variáveis categóricas restantes, agruparemos aquelas categorias
#  que estão presentes em menos de 1% das observações. Ou seja, todos
#  os valores de variáveis categóricas que são compartilhados por menos de 1% 
#  de as casas serão substituídos pela string "Rare".
qual_vars  = qual_vars + finish_vars + ['BsmtExposure','GarageFinish','Fence']

# Captura as variáveis categóricas restantes
# (aquelas que não remapeamos)
cat_others = [ var for var in cat_vars if var not in qual_vars ]


# RareLabelEncoder so funciona con tipo object
X_train['MSSubClass'] = X_train['MSSubClass'].astype('O')
X_test['MSSubClass'] = X_test['MSSubClass'].astype('O')


# configurar la clase
rare_encoder = RareLabelEncoder(tol=0.01, n_categories=1, variables=cat_others)
#  ajustar la clase al conjunto de entrenamiento
rare_encoder.fit(X_train)
# La clase aprende y almacena los parametros
rare_encoder.encoder_dict_

# aplicar
X_train = rare_encoder.transform(X_train)
X_test = rare_encoder.transform(X_test)

##################
### e- Variáveis categóricas: converter strings para números
# configurar la clase
cat_encoder = OrdinalEncoder(encoding_method='ordered', variables=cat_others)

#  ajustar la clase al conjunto de entrenamiento --> mappings
cat_encoder.fit(X_train, y_train)

# mappings son  almacenados
cat_encoder.encoder_dict_

# aplicar
X_train = cat_encoder.transform(X_train)
X_test = cat_encoder.transform(X_test)

# Verifica a ausência de NA no conjunto de train
# [var for var in X_train.columns if X_train[var].isnull().sum() > 0]


# Analisar las variables
def analyse_vars(train, y_train, var):
    
    # function plots median house sale price per encoded
    # category
    
    tmp = pd.concat([X_train, np.log(y_train)], axis=1)
    
    tmp.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.ylim(2.2, 2.6)
    plt.ylabel('SalePrice')
    plt.show()
    
for var in cat_others:
    analyse_vars(X_train, y_train, var)
    
##################
### f- Colocar as variáveis em uma escala semelhante (modelos lineares)
# Criar scaler
scaler = MinMaxScaler()
# Ajusta (fit) o scaler ao conjunto de treino
scaler.fit(X_train)

# Transforma o conjunto de treino e teste

# O sklearn retorna arrays do numpy, então envolvemos o
# array com um dataframe do pandas
X_train = pd.DataFrame( scaler.transform(X_train),
                        columns = X_train.columns  )

X_test = pd.DataFrame(  scaler.transform(X_test),
                        columns = X_train.columns  )
 