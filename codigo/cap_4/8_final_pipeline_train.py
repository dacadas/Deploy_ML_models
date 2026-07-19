# data manipulation and plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
# for saving the pipeline
import joblib

# from Scikit-learn
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Binarizer

# from feature-engine
from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
    CategoricalImputer,
)

from feature_engine.encoding import (
    RareLabelEncoder,
    OrdinalEncoder,
)

from feature_engine.transformation import (
    LogTransformer,
    YeoJohnsonTransformer,
)

from feature_engine.selection import DropFeatures
from feature_engine.wrappers import SklearnTransformerWrapper

import preprocessar_dados_clase as pp

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# Para fins de reprodutibilidade
semilla  = 0
np.random.seed( semilla )
random.seed(semilla) 


#### Carregar o dataset
path_data = 'D:/Diego/Curso Udemy/Deploy_ML_models/dataset/'

data = pd.read_csv(path_data + 'train.csv')
print(data.shape)

## Elimina a coluna ID que não traz informação
data.drop('Id', axis=1, inplace=True)

# Cast MSSubClass as object
data['MSSubClass'] = data['MSSubClass'].astype('O')

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


for df in (X_train, X_test):
    df["MSSubClass"] = df["MSSubClass"].astype("object")
    

#############################################################
#### Engenharia de dados --

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA_FREQUENT = ['BsmtQual', 'BsmtExposure',
                                     'BsmtFinType1', 'GarageFinish']


CATEGORICAL_VARS_WITH_NA_MISSING = ['FireplaceQu']


# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ['LotFrontage']


TEMPORAL_VARS = ['YearRemodAdd']
REF_VAR = "YrSold"

# this variable is to calculate the temporal variable,
# can be dropped afterwards
DROP_FEATURES = ["YrSold"]

# variables to log transform
NUMERICALS_LOG_VARS = ["LotFrontage", "1stFlrSF", "GrLivArea"]


# variables to binarize
BINARIZE_VARS = ['ScreenPorch']

# variables to map
QUAL_VARS = ['ExterQual', 'BsmtQual',
             'HeatingQC', 'KitchenQual', 'FireplaceQu']

EXPOSURE_VARS = ['BsmtExposure']

FINISH_VARS = ['BsmtFinType1']

GARAGE_VARS = ['GarageFinish']

FENCE_VARS = ['Fence']


# categorical variables to encode
CATEGORICAL_VARS = ['MSSubClass',  'MSZoning',  'LotShape',  'LandContour',
                    'LotConfig', 'Neighborhood', 'RoofStyle', 'Exterior1st',
                    'Foundation', 'CentralAir', 'Functional', 'PavedDrive',
                    'SaleCondition']


# variable mappings
QUAL_MAPPINGS = {'Po': 1, 'Fa': 2, 'TA': 3,
                 'Gd': 4, 'Ex': 5, 'Missing': 0, 'NA': 0}

EXPOSURE_MAPPINGS = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

FINISH_MAPPINGS = {'Missing': 0, 'NA': 0, 'Unf': 1,
                   'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

GARAGE_MAPPINGS = {'Missing': 0, 'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}


# the selected variables
FEATURES = [
    'MSSubClass',
    'MSZoning',
    'LotFrontage',
    'LotShape',
    'LandContour',
    'LotConfig',
    'Neighborhood',
    'OverallQual',
    'OverallCond',
    'YearRemodAdd',
    'RoofStyle',
    'Exterior1st',
    'ExterQual',
    'Foundation',
    'BsmtQual',
    'BsmtExposure',
    'BsmtFinType1',
    'HeatingQC',
    'CentralAir',
    '1stFlrSF',
    '2ndFlrSF',
    'GrLivArea',
    'BsmtFullBath',
    'HalfBath',
    'KitchenQual',
    'TotRmsAbvGrd',
    'Functional',
    'Fireplaces',
    'FireplaceQu',
    'GarageFinish',
    'GarageCars',
    'GarageArea',
    'PavedDrive',
    'WoodDeckSF',
    'ScreenPorch',
    'SaleCondition',
    # this one is only to calculate temporal variable:
    "YrSold",
]
    
    
X_train = X_train[FEATURES]
X_test = X_test[FEATURES]


# set up the pipeline
price_pipe = Pipeline([

    # ===== IMPUTATION =====
    # impute categorical variables with string missing
    ('missing_imputation', CategoricalImputer(
        imputation_method='missing', variables=CATEGORICAL_VARS_WITH_NA_MISSING)),

    ('frequent_imputation', CategoricalImputer(
        imputation_method='frequent', variables=CATEGORICAL_VARS_WITH_NA_FREQUENT)),

    # add missing indicator
    ('missing_indicator', AddMissingIndicator(variables=NUMERICAL_VARS_WITH_NA)),

    # impute numerical variables with the mean
    ('mean_imputation', MeanMedianImputer(
        imputation_method='mean', variables=NUMERICAL_VARS_WITH_NA
    )),
    
    
    # == TEMPORAL VARIABLES ====
    ('elapsed_time', pp.TemporalvariableTransformer(
        variables=TEMPORAL_VARS, reference_variable=REF_VAR)),

    ('drop_features', DropFeatures(features_to_drop=[REF_VAR])),

   

    # ==== VARIABLE TRANSFORMATION =====
    ('log', LogTransformer(variables=NUMERICALS_LOG_VARS)),
    
#    ('yeojohnson', YeoJohnsonTransformer(variables=NUMERICALS_YEO_VARS)),
    
    ('binarizer', SklearnTransformerWrapper(
        transformer=Binarizer(threshold=0), variables=BINARIZE_VARS)),
    

    # === mappers ===
    ('mapper_qual', pp.Mapper(
        variables=QUAL_VARS, mappings=QUAL_MAPPINGS)),

    ('mapper_exposure', pp.Mapper(
        variables=EXPOSURE_VARS, mappings=EXPOSURE_MAPPINGS)),

    ('mapper_finish', pp.Mapper(
        variables=FINISH_VARS, mappings=FINISH_MAPPINGS)),

    ('mapper_garage', pp.Mapper(
        variables=GARAGE_VARS, mappings=GARAGE_MAPPINGS)),
    
#    ('mapper_fence', pp.Mapper(
#        variables=FENCE_VARS, mappings=FENCE_MAPPINGS)),


    # == CATEGORICAL ENCODING
    # Corrigido: adicionado ignore_format=True para evitar o erro de tipagem com o MSSubClass
    ('rare_label_encoder', RareLabelEncoder(
        tol=0.01, n_categories=1, variables=CATEGORICAL_VARS, ignore_format=True
    )),

    # encode categorical and discrete variables using the target mean
    # Corrigido: adicionado ignore_format=True
    ('categorical_encoder', OrdinalEncoder(
        encoding_method='ordered', variables=CATEGORICAL_VARS, ignore_format=True)),
    
    # Removido: 'mssubclass_encoder' era redundante pois o MSSubClass já foi tratado acima
    
    ('scaler', MinMaxScaler()),
#    ('selector', SelectFromModel(Lasso(alpha=0.001, random_state=0))),
    ('Lasso', Lasso(alpha=0.001, random_state=0)),
])


# train the pipeline
price_pipe.fit(X_train, y_train)


# evaluate the model:
# ====================
# 1. Conjunto de Treino
pred_train = price_pipe.predict(X_train)

print('--- TREINO ---')
print('train mse: {}'.format(int(
    mean_squared_error(np.exp(y_train), np.exp(pred_train)))))

# CORRIGIDO: Tiramos o squared=False e envolvemos no np.sqrt()
print('train rmse: {}'.format(int(
    np.sqrt(mean_squared_error(np.exp(y_train), np.exp(pred_train))))))

print('train r2: {}'.format( 
    r2_score(np.exp(y_train), np.exp(pred_train))))


print('\n--- TESTE ---')
# 2. Conjunto de Teste
pred_test = price_pipe.predict(X_test)

print('test mse: {}'.format(int(
    mean_squared_error(np.exp(y_test), np.exp(pred_test)))))

# CORRIGIDO: Tiramos o squared=False e envolvemos no np.sqrt()
print('test rmse: {}'.format(int(
    np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred_test))))))

print('test r2: {}'.format(
    r2_score(np.exp(y_test), np.exp(pred_test))))

print('\nAverage house price: ', int(np.exp(y_train).median()))



# let's evaluate our predictions respect to the real sale price
plt.figure()
plt.scatter(y_test, price_pipe.predict(X_test))
plt.xlabel('True House Price')
plt.ylabel('Predicted House Price')
plt.title('Evaluation of Lasso Predictions')

# let's evaluate the distribution of the errors: 
# they should be fairly normally distributed
plt.figure()
y_test.reset_index(drop=True, inplace=True)

preds = pd.Series(price_pipe.predict(X_test))

errors = y_test - preds
errors.hist(bins=30)
plt.show()



# now let's save the scaler
joblib.dump(price_pipe, path_data + 'price_pipe.joblib') 

##############################################################
##############################################################
# Carregar novos dados

data = pd.read_csv(path_data + 'test.csv')
print(data.shape)

## Elimina a coluna ID que não traz informação
data.drop('Id', axis=1, inplace=True)

# Cast MSSubClass as object
data['MSSubClass'] = data['MSSubClass'].astype('O')

data = data[FEATURES]

print(data.shape)


new_vars_with_na = [
    var for var in FEATURES
    if var not in CATEGORICAL_VARS_WITH_NA_FREQUENT +
    CATEGORICAL_VARS_WITH_NA_MISSING +
    NUMERICAL_VARS_WITH_NA
    and data[var].isnull().sum() > 0]

# new_vars_with_na


# data[new_vars_with_na].isnull().mean()

data.dropna(subset=new_vars_with_na, inplace=True)

print(data.shape)


new_preds = price_pipe.predict(data)
# let's plot the predicted sale prices
pd.Series(np.exp(new_preds)).hist(bins=50)