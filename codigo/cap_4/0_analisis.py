# Para manipular datasets
import pandas as pd
import numpy as np

# Para plotagem de gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# Para a transformação Yeo-Johnson
import scipy.stats as stats

# Para exibir todas as colunas do dataframe no notebook
pd.pandas.set_option('display.max_columns', None)



#### Carregar o dataset
path_data = 'D:/Diego/Curso Udemy/Deploy_ML_models/dataset/'

data = pd.read_csv(path_data + 'train.csv')
print(data.shape)


## Elimina a coluna ID que não traz informação
data.drop('Id', axis=1, inplace=True)

# Há 79 variáveis e uma de preço (80 colunas)


###############################################################################
####            Exploração de dados
###############################################################################    
# ### a) Variável alvo -- verificar tipo de distribuição
# data['SalePrice'].hist(bins=50, density=True)
# plt.ylabel('Number of houses')
# plt.xlabel('Sale Price')
# plt.show()

# # Como tem uma distribuição assimétrica (skewed), transformaremos em log para torná-la gaussiana
# np.log(data['SalePrice']).hist(bins=50, density=True)
# plt.ylabel('Number of houses')
# plt.xlabel('Log of Sale Price')
# plt.show()



###############################################################################    
### b) Identificar tipo de variáveis
## b.1) Categóricas?
cat_vars = [var for var in data.columns if data[var].dtype == 'O']

# Vamos adicionar MSSubClass à lista de variáveis categóricas
cat_vars = cat_vars + ['MSSubClass']

# Deixar todas as variáveis categóricas como objetos (Object)
data[cat_vars] = data[cat_vars].astype('O')

## b.2) Numéricas?
num_vars = [var for var in data.columns if var not in cat_vars and var != 'SalePrice']

###############################################################################    
### c) O que fazer com dados faltantes? (Missing values)
# Identificar a porcentagem de valores faltantes
vars_with_na = [var for var in data.columns if data[var].isnull().sum() > 0]
data[vars_with_na].isnull().mean().sort_values(ascending=False)
# Gráfico
data[vars_with_na].isnull().mean().sort_values(
    ascending=False).plot.bar(figsize=(10, 4))
plt.ylabel('Percentage of missing data')
plt.axhline(y=0.90, color='r', linestyle='-')
plt.axhline(y=0.80, color='g', linestyle='-')
plt.show()


## c.1) Separar quais categóricas e quais numéricas têm informações faltantes
cat_na = [var for var in cat_vars if var in vars_with_na]
num_na = [var for var in num_vars if var in vars_with_na]

print('N de var. categoricas con nan: ', len(cat_na))
print('N de var. numericas con na: ', len(num_na))

## c.2) Antes de tomar qualquer decisão em relação às variáveis com dados 
#       faltantes, é necessário encontrar a relação delas com a variável a ser predita
def analyse_na_value(df, var):
    df = df.copy()
    # Observação estava faltando (1) ou não (0)
    df[var] = np.where(df[var].isnull(), 1, 0)

    # Vamos comparar a mediana do SalePrice nas observações onde o dado está faltando
    # vs as observações onde o dado está disponível

    # Determina o preço médio/mediana nos grupos 1 e 0,
    # e o desvio padrão do preço de venda,
    # e capturamos os resultados em um dataset temporário
    tmp = df.groupby(var)['SalePrice'].agg(['mean', 'std'])

    # Plota em um gráfico de barras
    tmp.plot(kind="barh", y="mean", legend=False,
             xerr="std", title="Sale Price", color='green')

    plt.show()

''' Em algumas variáveis, el preço médio de venda das casas para as quais
    falta informação difere do preço médio de venda daquelas para as 
    quais existe informação.
    Isso sugere que a ausência de dados pode ser um bom preditor do 
    preço de venda.
'''
for var in vars_with_na:
    analyse_na_value(data, var)
    
###############################################################################    
### d) Analisar variáveis numéricas

##########################################
## d.1) Temporais?
year_vars = [var for var in num_vars if 'Yr' in var or 'Year' in var]    
for var in year_vars:
    print(var, data[var].unique())
    print()
    
# Agrupar variáveis temporais e ver o comportamento
data.groupby('YrSold')['SalePrice'].median().plot()
plt.ylabel('Median House Price')

data.groupby('YearBuilt')['SalePrice'].median().plot()
plt.ylabel('Median House Price')

data.groupby('YearRemodAdd')['SalePrice'].median().plot()
plt.ylabel('Median House Price')

data.groupby('GarageYrBlt')['SalePrice'].median().plot()
plt.ylabel('Median House Price')

# Criar uma função que relacione essas variáveis temporais para tirar conclusões abrangentes
def analyse_year_vars(df, var):    
    df = df.copy()    
    # Captura a diferença entre uma variável de ano e o ano
    # em que a casa foi vendida
    df[var] = df['YrSold'] - df[var]
    
    df.groupby('YrSold')[var].median().plot()
    plt.ylabel('Time from ' + var)
    plt.show()
       
for var in year_vars:
    if var !='YrSold':
        analyse_year_vars(data, var)

## Ver a tendência da variável predita vs variável de tempo
def analyse_year_vars(df, var):
    
    df = df.copy()    
    # Captura a diferença entre uma variável de ano e o ano
    # em que a casa foi vendida
    df[var] = df['YrSold'] - df[var]
    
    plt.scatter(df[var], df['SalePrice'])
    plt.ylabel('SalePrice')
    plt.xlabel(var)
    plt.show()
        
for var in year_vars:
    if var !='YrSold':
        analyse_year_vars(data, var)
        
##########################################
## d.2) Discretas     
discrete_vars = [var for var in num_vars if len(data[var].unique()) < 20 and var not in year_vars]
print('Number of discrete variables: ', len(discrete_vars))   

## Plotar relações de 'categorias' vs variável a ser predita
for var in discrete_vars:
    # Faz o boxplot com Catplot
    sns.catplot(x=var, y='SalePrice', data=data, kind="box", height=4, aspect=1.5)
    # Adiciona os pontos de dados ao boxplot com stripplot
    sns.stripplot(x=var, y='SalePrice', data=data, jitter=0.1, alpha=0.3, color='k')
    plt.show()
    
##########################################
## d.3) Contínuas     
cont_vars = [ var for var in num_vars if var not in discrete_vars+year_vars]

## Plotar relações de contínuas (histograma vs variável a ser predita)
data[cont_vars].hist(bins=40, figsize=(15,15))
plt.show()

## Separar variáveis muito assimétricas (identificadas no passo anterior)
skewed = [ 'BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch',
          '3SsnPorch', 'ScreenPorch', 'MiscVal']

cont_vars = [ 'LotFrontage',    'LotArea',    'MasVnrArea',    'BsmtFinSF1',
                'BsmtUnfSF',    'TotalBsmtSF',    '1stFlrSF',    '2ndFlrSF',
                'GrLivArea',    'GarageArea',    'WoodDeckSF',    'OpenPorchSF',]
# Aplicar transformação Yeo-Johnson e plotar para posterior análise
# Cópia temporária dos dados
tmp = data.copy()

tmp[cont_vars] = tmp[cont_vars].astype('float64')
for var in cont_vars:
    # Filtra apenas os valores que são válidos (finitos)
    valid_data = data[var][data[var].notna()]
    
    if len(valid_data) > 0:
        # Transforma apenas os valores válidos
        transformed, param = stats.yeojohnson(valid_data)
        
        # Aloca os valores transformados de volta nas posições corretas
        tmp.loc[valid_data.index, var] = transformed
    
# Plota os histogramas das variáveis transformadas
tmp[cont_vars].hist(bins=30, figsize=(15,15))
plt.show()

# Não garante, por isso é preciso analisar e separar as variáveis que têm uma distribuição mais parecida com a gaussiana
## Plotar relação das variáveis antes e depois da transformação com a variável predita
for var in cont_vars:
    
    plt.figure(figsize=(12,4))
    
    # Plota a variável original vs o preço de venda    
    plt.subplot(1, 2, 1)
    plt.scatter(data[var], np.log(data['SalePrice']))
    plt.ylabel('Sale Price')
    plt.xlabel('Original ' + var)

    # Plota a variável transformada vs o preço de venda
    plt.subplot(1, 2, 2)
    plt.scatter(tmp[var], np.log(tmp['SalePrice']))
    plt.ylabel('Sale Price')
    plt.xlabel('Transformed ' + var)
                
    plt.show()
   
## Antes de aplicar transformações que utilizem logaritmos, verificar valor zero
tmp = data.copy()
for var in ["LotFrontage", "1stFlrSF", "GrLivArea"]:
    # Transforma a variável com logaritmo
    tmp[var] = np.log(data[var])
    
tmp[["LotFrontage", "1stFlrSF", "GrLivArea"]].hist(bins=30)
plt.show()

for var in ["LotFrontage", "1stFlrSF", "GrLivArea"]:
    
    plt.figure(figsize=(12,4))
    
    # Plota a variável original vs o preço de venda    
    plt.subplot(1, 2, 1)
    plt.scatter(data[var], np.log(data['SalePrice']))
    plt.ylabel('Sale Price')
    plt.xlabel('Original ' + var)

    # Plota a variável transformada vs o preço de venda
    plt.subplot(1, 2, 2)
    plt.scatter(tmp[var], np.log(tmp['SalePrice']))
    plt.ylabel('Sale Price')
    plt.xlabel('Transformed ' + var)
                
    plt.show()


## E as que possuem distribuição tipo Rayleigh (assimétricas)?
# Transforma em binárias e verifica a tendência da variável predita
for var in skewed:
    
    tmp = data.copy()
    
    # Mapeia os valores da variável em 0 e 1
    tmp[var] = np.where(data[var]==0, 0, 1)
    
    # Determina o preço médio de venda nos valores mapeados
    tmp = tmp.groupby(var)['SalePrice'].agg(['mean', 'std'])

    # Plota em um gráfico de barras
    tmp.plot(kind="barh", y="mean", legend=False,
             xerr="std", title="Sale Price", color='green')

    plt.show()
''' Parece haver uma diferença no preço de venda entre os valores mapeados,
 mas os intervalos de confiança se sobrepõem; portanto, o mais provável é
 que essa diferença não seja significativa nem preditiva.'''
 
 
 
##########################################
## d.4) Categórias  
## Cardinalidade (baixa cardinalidade significa poucos rótulos/labels diferentes)
# Avalie quantas categorias diferentes estão presentes em cada uma de las variáveis.
data[cat_vars].nunique().sort_values(ascending=False).plot.bar(figsize=(12,5))

# Como lidar com cardinalidade???

## Qualidade das variáveis
''' Existem várias variáveis que fazem referência à qualidade de algum aspecto 
    da casa, como por exemplo a garagem, a cerca ou a cozinha. Substituirei estas
    categorias por números que aumentam em função da qualidade do espaço ou do
    cômodo.'''
# Fazer uma transformação por valores conhecidos para determinar a qualidade (visualmente) 
# é uma boa opção        
qual_mappings = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'Missing': 0, 'NA': 0}

qual_vars = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
             'HeatingQC', 'KitchenQual', 'FireplaceQu',
             'GarageQual', 'GarageCond',
            ]
for var in qual_vars:
    data[var] = data[var].map(qual_mappings)

exposure_mappings = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4, 'Missing': 0, 'NA': 0}
var = 'BsmtExposure'
data[var] = data[var].map(exposure_mappings)

finish_mappings = {'Missing': 0, 'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
finish_vars = ['BsmtFinType1', 'BsmtFinType2']
for var in finish_vars:
    data[var] = data[var].map(finish_mappings)
    
garage_mappings = {'Missing': 0, 'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
var = 'GarageFinish'
data[var] = data[var].map(garage_mappings)

fence_mappings = {'Missing': 0, 'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
var = 'Fence'
data[var] = data[var].map(fence_mappings)

qual_vars  = qual_vars + finish_vars + ['BsmtExposure','GarageFinish','Fence']


for var in qual_vars:
    # Faz o boxplot com Catplot
    sns.catplot(x=var, y='SalePrice', data=data, kind="box", height=4, aspect=1.5)
    # Adiciona os pontos de dados ao boxplot com stripplot
    sns.stripplot(x=var, y='SalePrice', data=data, jitter=0.1, alpha=0.3, color='k')
    plt.show()
        
    
## Labels que raramente aparecem -- podem prejudicar o modelo
cat_others = [    var for var in cat_vars if var not in qual_vars ]

def analyse_rare_labels(df, var, rare_perc):
    df = df.copy()

    # Determina a % de observações por categoria
    tmp = df.groupby(var)['SalePrice'].count() / len(df)

    # Retorna as categorias que são raras
    return tmp[tmp < rare_perc]

# Exibe as categorias que estão presentes em menos de
# 1% das observações
for var in cat_others:
    print(analyse_rare_labels(data, var, 0.01))
    print()
    
    
# Algumas das variáveis categóricas mostram múltiplas etiquetas que estão 
# presentes em menos de 1% das casas.
# As etiquetas que estão sub-representadas no conjunto de dados tendem a 
# causar um ajuste excessivo (overfitting) dos modelos de aprendizado de máquina.
# Por isso queremos eliminá-las.

# Plotar e analisar
for var in cat_others:
    # Faz o boxplot com Catplot
    sns.catplot(x=var, y='SalePrice', data=data, kind="box", height=4, aspect=1.5)
    # Adiciona os pontos de dados ao boxplot com stripplot
    sns.stripplot(x=var, y='SalePrice', data=data, jitter=0.1, alpha=0.3, color='k')
    plt.show()