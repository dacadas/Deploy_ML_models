from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class FirstLetterExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.var = variables
    def fit(self, X, y=None):    
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.var :
            X[feature] = X[feature].str[0]
        return X

class DataFrameStandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        self.variables = variables
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.variables])
        return self

    def transform(self, X):
        X = X.copy()

        X[self.variables] = self.scaler.transform(
            X[self.variables]
        )

        return X





# to handle datasets
import random
import pandas as pd
import numpy as np
# to divide train and test set
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# feature scaling


# from feature-engine
from feature_engine.imputation import (
    AddMissingIndicator,  # crea columna para indicar cuando falta algun valor
    MeanMedianImputer,    # imputar media o mediana
    CategoricalImputer,   # imputar mas frecuente ou colocar 'missing'
)
from feature_engine.encoding import (
    RareLabelEncoder,     # remover labels raros
)
from feature_engine.encoding import OneHotEncoder

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models
# from sklearn.metrics import accuracy_score, roc_auc_score


from sklearn.pipeline import Pipeline

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

# Para fins de reprodutibilidade
semilla  = 42
np.random.seed( semilla )
random.seed(semilla) 


#### Carregar o dataset
# data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')

path_data = 'D:/Diego/Curso Udemy/Deploy_ML_models/dataset/Titanic/'

data = pd.read_csv(path_data + 'titanic_completo_procesado.csv')

## Separar los datos 
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('survived', axis=1),  # predictors
    data['survived'],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=semilla)  # seed to ensure reproducibility

target = 'survived'
##################################################


# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA_FREQUENT = ['cabin', 'embarked']

CATEGORICAL_VARS_WITH_NA_MISSING = ['sex', 'cabin', 'embarked', 'title']

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_VARS_WITH_NA = ['age', 'fare']

NUMERICAL_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare']


FEATURES = ['sex', 'cabin', 'embarked', 
            'title', 'pclass', 'age', 
            'sibsp', 'parch', 'fare']

# set up the pipeline
surv_pipe = Pipeline([
    
    ("cabin", FirstLetterExtractor(variables = ['cabin'])),
    
    # ===== IMPUTATION =====
    # impute categorical variables with string missing
    ('missing_imputation', CategoricalImputer(
        imputation_method='missing', 
        variables=CATEGORICAL_VARS_WITH_NA_MISSING)),

    # add missing indicator
    ('missing_indicator', AddMissingIndicator(variables=NUMERICAL_VARS_WITH_NA)),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(
        imputation_method='median', 
        variables=NUMERICAL_VARS_WITH_NA
    )),
    
    ("scaler", DataFrameStandardScaler(NUMERICAL_VARS)),
    
    ('rare_label_encoder', RareLabelEncoder(
        tol=0.05, 
        n_categories=1, 
        variables=CATEGORICAL_VARS, 
        ignore_format=True
    )),
    
    ("onehot", OneHotEncoder(
        variables=CATEGORICAL_VARS,
        drop_last=True   # equivalente ao drop_first=True do pandas
    )),
    
    
    ('modelo', LogisticRegression(
        C=0.0005, 
        random_state=0)),
      
    ])
    
# train the pipeline
surv_pipe.fit(X_train, y_train)

pred_train = surv_pipe.predict(X_train)



# Para Acurácia: Precisamos das predições de classe (0 ou 1)
class_pred_train =  surv_pipe.predict(X_train)
class_pred_test  =  surv_pipe.predict(X_test)

# Para ROC-AUC: Precisamos das probabilidades da classe 1 (sobreviveu)
prob_pred_train = surv_pipe.predict_proba(X_train)[:, 1]
prob_pred_test  = surv_pipe.predict_proba(X_test)[:, 1]

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

