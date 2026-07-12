# Para manipular datasets
import os
import pandas as pd
import numpy as np
import random
# Para plotagem de gráficos
import matplotlib.pyplot as plt

# Para a transformação Yeo-Johnson
import scipy.stats as stats

# Para dividir o conjunto de treino e teste
from sklearn.model_selection import train_test_split

# Redimensionamento de atributos (Feature scaling)
from sklearn.preprocessing import MinMaxScaler

# Para salvar a classe do scaler treinada
import joblib

# Para visualizar todas as colunas no dataframe
pd.pandas.set_option('display.max_columns', None)



def criar_pasta(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    
# Para fins de reprodutibilidade
semilla  = 13
np.random.seed( semilla )
random.seed(semilla) 


#### Carregar o dataset
path_data = 'D:/Diego/Curso Udemy/Deploy_ML_models/dataset/'
path_out  = path_data + 'dado_preprocesado/'


criar_pasta(path_out)



data = pd.read_csv(path_data + 'train.csv')
print(data.shape)


## Elimina a coluna ID que não traz informação
data.drop('Id', axis=1, inplace=True)

###############################################################################
## Separa em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split( data.drop(['SalePrice'], axis=1), # Variáveis preditivas (X)
                                                     data['SalePrice'], # Alvo (y)
                                                     test_size    = 0.1, # Proporção do dataset a ser alocada para o conjunto de teste
                                                     random_state = semilla, # Estamos definindo a semente (seed) aqui
                                                    )
## Plotar o Y alvo
y_train.hist(bins=40, figsize=(15,15))
plt.show()
y_test.hist(bins=40, figsize=(15,15))
plt.show()

## Transformar o alvo para logaritmo para deixá-lo mais gaussiano
y_train = np.log(y_train)
y_test = np.log(y_test)
y_train.hist(bins=40, figsize=(15,15))
plt.show()
y_test.hist(bins=40, figsize=(15,15))
plt.show()


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

X_train[with_string_missing] = X_train[with_string_missing].fillna('Missing')
X_test[with_string_missing]  = X_test[with_string_missing].fillna('Missing')

# Variáveis para imputar com a categoria mais frequente
with_frequent_category = [ var for var in cat_vars_with_na if X_train[var].isnull().mean() < 0.1]

for var in with_frequent_category:    
    # Pode haver mais de 1 moda em uma variável
    # Pegamos a primeira com [0]    
    mode = X_train[var].mode()[0]
    print(var, mode)
    X_train[var] = X_train[var].fillna(mode)
    X_test[var] = X_test[var].fillna(mode)

# Verificar se há NaN
X_train[cat_vars_with_na].isnull().sum()
# [var for var in cat_vars_with_na if X_test[var].isnull().sum() > 0]



## a2 -- Numéricas 
# Primeiro identificar quais variáveis são numéricas
num_vars = [    var for var in X_train.columns if var not in cat_vars and var != 'SalePrice' ]

# Verificar dados faltantes
vars_with_na = [    var for var in num_vars    if X_train[var].isnull().sum() > 0]

# Exibe a porcentagem de valores faltantes por variável
X_train[vars_with_na].isnull().mean().sort_values(ascending=False)

# Substituir valores faltantes pela média no treino e teste e criar uma nova variável binária para indicar onde houve a mudança
for var in vars_with_na:

    # Calcula a média usando o conjunto de treino
    mean_val = X_train[var].mean()    
    print(var, mean_val)

    # Adiciona o indicador binário de ausência (no treino e no teste)
    X_train[var + '_na'] = np.where(X_train[var].isnull(), 1, 0)
    X_test[var + '_na'] = np.where(X_test[var].isnull(), 1, 0)

    # Substitui os valores faltantes pela média (no treino e no teste)
    X_train[var] = X_train[var].fillna(mean_val)
    X_test[var] = X_test[var].fillna(mean_val)

# Verifica se não temos mais valores faltantes nas variáveis modificadas
X_train[vars_with_na].isnull().sum()
[var for var in vars_with_na if X_test[var].isnull().sum() > 0]



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

X_train.drop(['YrSold'], axis=1, inplace=True)
X_test.drop(['YrSold'], axis=1, inplace=True)


##################
### c- Variáveis com distribuição não gaussiana (aplicar transformação,
# seja log, seja Yeo-Johnson para deixá-las mais gaussianas)
# Especificamente aqui, na análise exploratória vimos que o log ajuda
for var in ["LotFrontage", "1stFlrSF", "GrLivArea"]:
    X_train[var] = np.log(X_train[var])
    X_test[var]  = np.log(X_test[var])
# [var for var in ["LotFrontage", "1stFlrSF", "GrLivArea"] if X_test[var].isnull().sum() > 0]
# [var for var in ["LotFrontage", "1stFlrSF", "GrLivArea"] if X_train[var].isnull().sum() > 0]



# A transformação Yeo-Johnson descobre o melhor expoente para transformar a variável
# Ela precisa aprender isso a partir do conjunto de treino: 
X_train['LotArea'], param = stats.yeojohnson(X_train['LotArea'])

# E então aplica a transformação ao conjunto de teste com o mesmo
# parâmetro: veja como desta vez passamos 'param' como argumento para a
# função yeojohnson
X_test['LotArea'] = stats.yeojohnson(X_test['LotArea'], lmbda=param)


# Verifica a ausência de NA no conjunto de teste
[var for var in X_train.columns if X_test[var].isnull().sum() > 0]
[var for var in X_train.columns if X_train[var].isnull().sum() > 0]


# Havia algumas variáveis muito assimétricas, vamos transformá-las em variáveis binárias.
skewed = [
    'BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch',
    '3SsnPorch', 'ScreenPorch', 'MiscVal'
]
for var in skewed:   
    # Mapeia os valores da variável em 0 e 1
    X_train[var] = np.where(X_train[var]==0, 0, 1)
    X_test[var] = np.where(X_test[var]==0, 0, 1)



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

# Verifica a ausência de NA no conjunto de treino
[var for var in X_train.columns if X_train[var].isnull().sum() > 0]

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

def find_frequent_labels(df, var, rare_perc):    
    # A função encontra os rótulos que são compartilhados por mais de
    # uma determinada % de as casas no dataset
    df = df.copy()
    tmp = df.groupby(var)[var].count() / len(df)
    return tmp[tmp > rare_perc].index

for var in cat_others:
    # Encontra as categorias frequentes
    frequent_ls = find_frequent_labels(X_train, var, 0.01)    
    print(var, frequent_ls)
    print()    
    # Substitui categorias raras pela string "Rare"
    X_train[var] = np.where(X_train[var].isin(  frequent_ls), X_train[var], 'Rare')   
    X_test[var] = np.where(X_test[var].isin(    frequent_ls), X_test[var], 'Rare')
    
    
##################
### e- Variáveis categóricas: converter strings para números

# Esta função atribuirá valores discretos às strings das variáveis,
# de modo que o menor valor corresponda à categoria que possui
# o menor preço médio de venda da casa
def replace_categories(train, test, y_train, var, target): 
    tmp = pd.concat([X_train, y_train], axis=1)
    
    # Ordene as categorias de uma variável a partir daquela que possui o menor preço de venda
    # até a de maior valor
    ordered_labels = tmp.groupby([var])[target].mean().sort_values().index

    # Cria um dicionário de categorias ordenadas para valores inteiros
    ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
    
    print(var, ordinal_label)
    print()

    # Usa o dicionário para substituir as strings categóricas por inteiros
    train[var] = train[var].map(ordinal_label)
    test[var]  = test[var].map(ordinal_label)

for var in cat_others:
    replace_categories(X_train, X_test, y_train, var, 'SalePrice')


# Verifica a ausência de NA no conjunto de teste
# [var for var in X_test.columns if X_test[var].isnull().sum() > 0]
# [var for var in X_train.columns if X_train[var].isnull().sum() > 0]

# Permita-me mostrar o que quero dizer com relação monotônica entre
# as etiquetas e o alvo
def analyse_vars(train, y_train, var):
    
    # Função que plota a mediana do preço de venda por categoria codificada
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




## Para finalizar, salvar o novo conjunto
X_train.to_csv(path_out + 'xtrain.csv', index=False)
X_test.to_csv(path_out + 'xtest.csv', index=False)
y_train.to_csv(path_out + 'ytrain.csv', index=False)
y_test.to_csv(path_out + 'ytest.csv', index=False)


joblib.dump(scaler, path_out + 'minmax_scaler.joblib')