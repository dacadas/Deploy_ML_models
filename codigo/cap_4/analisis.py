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
#### Exploracion de datos

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




### b) identificar tipo de variables
## b.1) categoricas?
cat_vars = [var for var in data.columns if data[var].dtype == 'O']

# lets add MSSubClass to the list of categorical variables
cat_vars = cat_vars + ['MSSubClass']

# dejar todas llas variables categoricas como objetos
data[cat_vars] = data[cat_vars].astype('O')

## b.2) numericas?
num_vars = [var for var in data.columns if var not in cat_vars and var != 'SalePrice']


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