# to handle datasets
import pandas as pd
import numpy as np
import random

# for visualization
import matplotlib.pyplot as plt

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

# to persist the model and the scaler
import joblib

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

# Para fins de reprodutibilidade
semilla  = 42
np.random.seed( semilla )
random.seed(semilla) 




#### Carregar o dataset
# data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')

path_data = 'D:/Diego/Curso Udemy/Deploy_ML_models/dataset/Titanic/'



# data = pd.read_csv(path_data + 'data_ori.csv')

# ## preprocesar segun curso
# data = data.replace('?', np.nan)

# def get_first_cabin(row):
#     try:
#         return row.split()[0]
#     except:
#         return np.nan    
# data['cabin'] = data['cabin'].apply(get_first_cabin)

# # extracts the title (Mr, Ms, etc) from the name variable
# def get_title(passenger):
#     line = passenger
#     if re.search('Mrs', line):
#         return 'Mrs'
#     elif re.search('Mr', line):
#         return 'Mr'
#     elif re.search('Miss', line):
#         return 'Miss'
#     elif re.search('Master', line):
#         return 'Master'
#     else:
#         return 'Other'    
# data['title'] = data['name'].apply(get_title)

# # cast numerical variables as floats
# data['fare'] = data['fare'].astype('float')
# data['age'] = data['age'].astype('float')

# # drop unnecessary variables
# data.drop(labels=['name','ticket', 'boat', 'body','home.dest'], axis=1, inplace=True)
# data.to_csv(path_data + 'titanic_completo_procesado.csv', index=False)



data = pd.read_csv(path_data + 'titanic_completo_procesado.csv')

###############################################################################
#######  Exploracion de datos
target = 'survived'

### b) Identificar tipo de variáveis
## b.1) Categóricas?
vars_cat = [var for var in data.columns if data[var].dtype == 'O']
## b.2) Numéricas?
vars_num = [var for var in data.columns if var not in vars_cat and var != target]

print('Number of numerical variables: {}'.format(len(vars_num)))
print('Number of categorical variables: {}'.format(len(vars_cat)))


# Identificar a porcentagem de valores faltantes
vars_with_na = [var for var in data.columns if data[var].isnull().sum() > 0]
print( data[vars_with_na].isnull().mean().sort_values(ascending=False) )

vars_cat_with_na = [ var for var in vars_cat if data[var].isnull().sum() > 0]
vars_num_with_na = [ var for var in vars_num if data[var].isnull().sum() > 0]




## determinar cardenalidad de las categoricas
# Avalie quantas categorias diferentes estão presentes em cada uma de las variáveis.
conta = data[vars_cat].nunique().sort_values(ascending=False)
ax    = conta.plot.bar(figsize=(12,5))
ax.bar_label(ax.containers[0], fmt='%d', padding=3)
plt.tight_layout()
plt.show()


# Determine the distribution of numerical variables
data[vars_num].hist(bins=40, figsize=(15,15))
plt.show()


## Separar los datos 
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('survived', axis=1),  # predictors
    data['survived'],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=semilla)  # seed to ensure reproducibility

## Plotar o Y alvo
y_train.hist(bins=40, figsize=(15,15))
plt.show()
y_test.hist(bins=40, figsize=(15,15))
plt.show()

###############################################################################
#######  Engenieria de datos
## Extract only the letter (and drop the number) from the variable Cabin
X_train['cabin'] = X_train['cabin'].str[0]
X_test['cabin']  = X_test['cabin'].str[0]


## Fill in Missing data in numerical variables:
#  - Add a binary missing indicator
#  - Fill NA in original variable with the median 
for var in vars_num_with_na:
    # Calcula a mediana usando o conjunto de treino
    median_val = X_train[var].median()    
    print(var, median_val)

    # Adiciona o indicador binário de ausência (no treino e no teste)
    X_train[var + '_na'] = np.where(X_train[var].isnull(), 1, 0)
    X_test[var + '_na']  = np.where( X_test[var].isnull(), 1, 0)

    # Substitui os valores faltantes pela mediana (no treino e no teste)
    X_train[var] = X_train[var].fillna(median_val)
    X_test[var] = X_test[var].fillna(median_val)
    
## Replace Missing data in categorical variables with the string Missing    
X_train[vars_cat_with_na] = X_train[vars_cat_with_na].fillna('Missing')
X_test[vars_cat_with_na]  = X_test[vars_cat_with_na].fillna('Missing')
   
## remove labels present in less than 5 % of the passengers
def find_frequent_labels(df, var, rare_perc):    
    # A função encontra os rótulos que são compartilhados por mais de
    # uma determinada % de as casas no dataset
    df = df.copy()
    tmp = df.groupby(var)[var].count() / len(df)
    return tmp[tmp > rare_perc].index

for var in vars_cat:
    # Encontra as categorias frequentes
    frequent_ls = find_frequent_labels(X_train, var, 0.05)    
    print(var, frequent_ls)
    print()    
    # Substitui categorias raras pela string "Rare"
    X_train[var] = np.where(X_train[var].isin(  frequent_ls), X_train[var], 'Rare')   
    X_test[var]  = np.where( X_test[var].isin(  frequent_ls), X_test[var], 'Rare')
    
    
    
## Perform one hot encoding of categorical variables into k-1 binary variables    
  # *  k-1, means that if the variable contains 9 different categories, 
  #    we create 8 different binary variables
  # *  Remember to drop the original categorical variable (the one with 
  #    the strings) after the encoding  
## Perform one hot encoding of categorical variables into k-1 binary variables    
X_train_cat = pd.get_dummies(X_train[vars_cat], drop_first=True)  
X_test_cat  = pd.get_dummies(X_test[vars_cat], drop_first=True) 
X_test_cat  = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0) # CORREÇÃO OHE
  
X_train.drop(vars_cat, axis=1, inplace=True)  
X_test.drop(vars_cat, axis=1, inplace=True)  
  
## Scale the variables
scaler = StandardScaler()
scaler.fit(X_train[vars_num]) 

# CORREÇÃO DOS ÍNDICES
X_train_num = pd.DataFrame(scaler.transform(X_train[vars_num]), columns=vars_num, index=X_train.index)
X_test_num  = pd.DataFrame(scaler.transform(X_test[vars_num]), columns=vars_num, index=X_test.index)

X_train.drop(vars_num, axis=1, inplace=True)  
X_test.drop(vars_num, axis=1, inplace=True) 

# montar conjuntos (agora vai funcionar perfeitamente sem gerar NaNs)
X_train = pd.concat([X_train, X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test, X_test_num, X_test_cat], axis=1)

del X_train_num, X_train_cat,  X_test_num, X_test_cat

joblib.dump(scaler, 'StandardScaler.joblib')

################## entenamiento
## Train the Logistic Regression model
# Set the regularization parameter to 0.0005
# Set the seed to 0
lr_model     = LogisticRegression(C=0.0005, random_state=0)

# Treinar o modelo
lr_model.fit(X_train, y_train)

# Para Acurácia: Precisamos das predições de classe (0 ou 1)
class_pred_train = lr_model.predict(X_train)
class_pred_test  = lr_model.predict(X_test)

# Para ROC-AUC: Precisamos das probabilidades da classe 1 (sobreviveu)
prob_pred_train = lr_model.predict_proba(X_train)[:, 1]
prob_pred_test  = lr_model.predict_proba(X_test)[:, 1]



## Determine:
#     - accuracy 
#     - roc-auc 
# Important, remember that to determine the accuracy, you need the outcome 0, 1
# referring to survived or not. But to determine the roc-auc you need the 
# probability of survival.

auc_train =  accuracy_score( y_train, class_pred_train)
roc_auc_score_train =  roc_auc_score( y_train, prob_pred_train)

print('train auc: {}'.format((auc_train)))
print('train roc_auc_score: {}'.format((roc_auc_score_train)))
print()

# test
auc_test =  accuracy_score( y_test, class_pred_test)
roc_auc_score_test =  roc_auc_score( y_test, prob_pred_test)

print('test auc: {}'.format((roc_auc_score_test)))
print('test roc_auc_score: {}'.format((roc_auc_score_test)))
print()






