# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# for the yeo-johnson transformation
import scipy.stats as stats

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)



#### cargar dataset
path_data = 'D:/Diego/Curso Udemy/Deploy_ML_models/dataset/'

data = pd.read_csv(path_data + 'train.csv')
print(data.shape)


## Elimina columna ID que no dice nada
data.drop('Id', axis=1, inplace=True)

# hay 79 variables y una de precio (80 columnas)


###############################################################################
####            Exploracion de datos
###############################################################################    
# ### a) variable objetivo -- verificar tipo de distribucion
# data['SalePrice'].hist(bins=50, density=True)
# plt.ylabel('Number of houses')
# plt.xlabel('Sale Price')
# plt.show()

# # como tiene una distribucion sqwe, transformaremos en log para gaussiana
# np.log(data['SalePrice']).hist(bins=50, density=True)
# plt.ylabel('Number of houses')
# plt.xlabel('Log of Sale Price')
# plt.show()



###############################################################################    
### b) identificar tipo de variables
## b.1) categoricas?
cat_vars = [var for var in data.columns if data[var].dtype == 'O']

# lets add MSSubClass to the list of categorical variables
cat_vars = cat_vars + ['MSSubClass']

# dejar todas llas variables categoricas como objetos
data[cat_vars] = data[cat_vars].astype('O')

## b.2) numericas?
num_vars = [var for var in data.columns if var not in cat_vars and var != 'SalePrice']

###############################################################################    
### c) Que hacer con dados perdidos? missing values
# identificar el porcetanje de valores perdidos
vars_with_na = [var for var in data.columns if data[var].isnull().sum() > 0]
data[vars_with_na].isnull().mean().sort_values(ascending=False)
# plot
data[vars_with_na].isnull().mean().sort_values(
    ascending=False).plot.bar(figsize=(10, 4))
plt.ylabel('Percentage of missing data')
plt.axhline(y=0.90, color='r', linestyle='-')
plt.axhline(y=0.80, color='g', linestyle='-')
plt.show()


## c.1) separar quales categoricas e cuales numericas tienen missing information
cat_na = [var for var in cat_vars if var in vars_with_na]
num_na = [var for var in num_vars if var in vars_with_na]

print('N de var. categoricas con nan: ', len(cat_na))
print('N de var. numericas con na: ', len(num_na))

## c.2) antes de tomar qualquer decision en relacion a las variables con datos 
#       faltantes, es necesario encontrar la relacion de ellas con la variable a ser predita
def analyse_na_value(df, var):
    df = df.copy()
    # observation was missing or 0 otherwise
    df[var] = np.where(df[var].isnull(), 1, 0)

    # let's compare the median SalePrice in the observations where data is missing
    # vs the observations where data is available

    # determine the median price in the groups 1 and 0,
    # and the standard deviation of the sale price,
    # and we capture the results in a temporary dataset
    tmp = df.groupby(var)['SalePrice'].agg(['mean', 'std'])

    # plot into a bar graph
    tmp.plot(kind="barh", y="mean", legend=False,
             xerr="std", title="Sale Price", color='green')

    plt.show()

''' En algunas variables, el precio de venta promedio de las casas para las que
    falta información difiere del precio de venta promedio de aquellas para las 
    que sí existe información.
    Esto sugiere que la ausencia de datos podría ser un buen predictor del 
    precio de venta.
'''
for var in vars_with_na:
    analyse_na_value(data, var)
    
###############################################################################    
### d) analisar variables numericas

##########################################
## d.1)temporales?
year_vars = [var for var in num_vars if 'Yr' in var or 'Year' in var]    
for var in year_vars:
    print(var, data[var].unique())
    print()
    
# Agrupar variables temporales y ver comportamiento
data.groupby('YrSold')['SalePrice'].median().plot()
plt.ylabel('Median House Price')

data.groupby('YearBuilt')['SalePrice'].median().plot()
plt.ylabel('Median House Price')

data.groupby('YearRemodAdd')['SalePrice'].median().plot()
plt.ylabel('Median House Price')

data.groupby('GarageYrBlt')['SalePrice'].median().plot()
plt.ylabel('Median House Price')

# criar uma função que relacione essas variaveis temporais para tirar conclusões abrangentes
def analyse_year_vars(df, var):    
    df = df.copy()    
    # capture difference between a year variable and year
    # in which the house was sold
    df[var] = df['YrSold'] - df[var]
    
    df.groupby('YrSold')[var].median().plot()
    plt.ylabel('Time from ' + var)
    plt.show()
       
for var in year_vars:
    if var !='YrSold':
        analyse_year_vars(data, var)

## ver tendencia variavel predida vs var tempo
def analyse_year_vars(df, var):
    
    df = df.copy()    
    # capture difference between a year variable and year
    # in which the house was sold
    df[var] = df['YrSold'] - df[var]
    
    plt.scatter(df[var], df['SalePrice'])
    plt.ylabel('SalePrice')
    plt.xlabel(var)
    plt.show()
        
for var in year_vars:
    if var !='YrSold':
        analyse_year_vars(data, var)
        
##########################################
## d.2) discretas     
discrete_vars = [var for var in num_vars if len(data[var].unique()) < 20 and var not in year_vars]
print('Number of discrete variables: ', len(discrete_vars))   

## plotar relaciones de 'categorias' vs variable a ser predita
for var in discrete_vars:
    # make boxplot with Catplot
    sns.catplot(x=var, y='SalePrice', data=data, kind="box", height=4, aspect=1.5)
    # add data points to boxplot with stripplot
    sns.stripplot(x=var, y='SalePrice', data=data, jitter=0.1, alpha=0.3, color='k')
    plt.show()
    
##########################################
## d3) continuas     
cont_vars = [ var for var in num_vars if var not in discrete_vars+year_vars]

## plotar relaciones de continuas histograma vs variable a ser predita
data[cont_vars].hist(bins=40, figsize=(15,15))
plt.show()

## separar variables muy asimetricas (identificadas en paso anterior)
skewed = [ 'BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch',
          '3SsnPorch', 'ScreenPorch', 'MiscVal']

cont_vars = [ 'LotFrontage',    'LotArea',    'MasVnrArea',    'BsmtFinSF1',
                'BsmtUnfSF',    'TotalBsmtSF',    '1stFlrSF',    '2ndFlrSF',
                'GrLivArea',    'GarageArea',    'WoodDeckSF',    'OpenPorchSF',]
# aplicar transformacion YEo-Jonshon e plotar para posterior analisi
# temporary copy of the data
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
    
# plot the histograms of the transformed variables
tmp[cont_vars].hist(bins=30, figsize=(15,15))
plt.show()

# no garantiza, por eso hay que analizar y separar las var que tienen una distribucion mas parecida a gaussiana
## plotar relacion de variables antes y despues de la transf con var_predita
for var in cont_vars:
    
    plt.figure(figsize=(12,4))
    
    # plot the original variable vs sale price    
    plt.subplot(1, 2, 1)
    plt.scatter(data[var], np.log(data['SalePrice']))
    plt.ylabel('Sale Price')
    plt.xlabel('Original ' + var)

    # plot transformed variable vs sale price
    plt.subplot(1, 2, 2)
    plt.scatter(tmp[var], np.log(tmp['SalePrice']))
    plt.ylabel('Sale Price')
    plt.xlabel('Transformed ' + var)
                
    plt.show()
   
## antes de aplicar transformaciones que utilicen logaritmos, verificar valor zero
tmp = data.copy()
for var in ["LotFrontage", "1stFlrSF", "GrLivArea"]:
    # transform the variable with logarithm
    tmp[var] = np.log(data[var])
    
tmp[["LotFrontage", "1stFlrSF", "GrLivArea"]].hist(bins=30)
plt.show()

for var in ["LotFrontage", "1stFlrSF", "GrLivArea"]:
    
    plt.figure(figsize=(12,4))
    
    # plot the original variable vs sale price    
    plt.subplot(1, 2, 1)
    plt.scatter(data[var], np.log(data['SalePrice']))
    plt.ylabel('Sale Price')
    plt.xlabel('Original ' + var)

    # plot transformed variable vs sale price
    plt.subplot(1, 2, 2)
    plt.scatter(tmp[var], np.log(tmp['SalePrice']))
    plt.ylabel('Sale Price')
    plt.xlabel('Transformed ' + var)
                
    plt.show()


## y las que son cons ditribucion tipo railegh (skewed)??
# transforma a binarias y verificar la tendencia de la variable pretia
for var in skewed:
    
    tmp = data.copy()
    
    # map the variable values into 0 and 1
    tmp[var] = np.where(data[var]==0, 0, 1)
    
    # determine mean sale price in the mapped values
    tmp = tmp.groupby(var)['SalePrice'].agg(['mean', 'std'])

    # plot into a bar graph
    tmp.plot(kind="barh", y="mean", legend=False,
             xerr="std", title="Sale Price", color='green')

    plt.show()
''' Parece haber una diferencia en el precio de venta entre los valores mapeados,
 pero los intervalos de confianza se solapan; por tanto, lo más probable es
 que dicha diferencia no sea significativa ni predictiva.'''
 
 
 
##########################################
## d.4) categoricas  
## cardinalidad  (baja cardinalidad es pocos labels diferentes)
# Evalúe cuántas categorías diferentes están presentes en cada una de las variables.
data[cat_vars].nunique().sort_values(ascending=False).plot.bar(figsize=(12,5))

# como lidar con cardinalidad???

## calidad de las variables
''' Existen varias variables que hacen referencia a la calidad de algún aspecto 
    de la casa, como por ejemplo el garaje, la cerca o la cocina. Sustituiré estas
    categorías por números que aumentan en función de la calidad del espacio o de
    la estancia.'''
# hacer una transformacion por valores conocidos para determinar calidad (visualmente) 
# es una buena opcion        
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
    # make boxplot with Catplot
    sns.catplot(x=var, y='SalePrice', data=data, kind="box", height=4, aspect=1.5)
    # add data points to boxplot with stripplot
    sns.stripplot(x=var, y='SalePrice', data=data, jitter=0.1, alpha=0.3, color='k')
    plt.show()
        
    
## labels q raramente aparecen -- pueden perjudicar el modelo
cat_others = [    var for var in cat_vars if var not in qual_vars ]

def analyse_rare_labels(df, var, rare_perc):
    df = df.copy()

    # determine the % of observations per category
    tmp = df.groupby(var)['SalePrice'].count() / len(df)

    # return categories that are rare
    return tmp[tmp < rare_perc]

# print categories that are present in less than
# 1 % of the observations
for var in cat_others:
    print(analyse_rare_labels(data, var, 0.01))
    print()
    
    
# Algunas de las variables categóricas muestran múltiples etiquetas que están 
# presentes en menos del 1% de las casas.
# Las etiquetas que están subrepresentadas en el conjunto de datos tienden a 
# causar un ajuste excesivo de los modelos de aprendizaje automático.
# Por eso queremos eliminarlos.

# plotar y analisar
for var in cat_others:
    # make boxplot with Catplot
    sns.catplot(x=var, y='SalePrice', data=data, kind="box", height=4, aspect=1.5)
    # add data points to boxplot with stripplot
    sns.stripplot(x=var, y='SalePrice', data=data, jitter=0.1, alpha=0.3, color='k')
    plt.show()

