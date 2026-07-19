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
#### Engenharia de dados -- ja sabemos quais variaves tem 'problemas'
# Valores faltantes conjunto de treino -- categorical 
CATEGORICAL_VARS_WITH_NA_FREQUENT = ['BsmtQual',
                                     'BsmtCond',
                                     'BsmtExposure',
                                     'BsmtFinType1',
                                     'BsmtFinType2',
                                     'Electrical',
                                     'GarageType',
                                     'GarageFinish',
                                     'GarageQual',
                                     'GarageCond']


CATEGORICAL_VARS_WITH_NA_MISSING = [
    'Alley', 'MasVnrType',  'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']


# Valores faltantes conjunto de treino -- numericas 
NUMERICAL_VARS_WITH_NA = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']


TEMPORAL_VARS = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
REF_VAR = "YrSold"


# variables to log transform
NUMERICALS_LOG_VARS = ["LotFrontage", "1stFlrSF", "GrLivArea"]

NUMERICALS_YEO_VARS = ['LotArea']


BINARIZE_VARS = [   'BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch',
                    '3SsnPorch', 'ScreenPorch', 'MiscVal'    ]

# variables to map
QUAL_VARS = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
             'HeatingQC', 'KitchenQual', 'FireplaceQu',
             'GarageQual', 'GarageCond',             ]

EXPOSURE_VARS = ['BsmtExposure']

FINISH_VARS = ['BsmtFinType1', 'BsmtFinType2']

GARAGE_VARS = ['GarageFinish']

FENCE_VARS = ['Fence']

# categorical variables to encode
CATEGORICAL_VARS = [
    'MSZoning',
    'Street',
    'Alley',
    'LotShape',
    'LandContour',
    'Utilities',
    'LotConfig',
    'LandSlope',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'Foundation',
    'Heating',
    'CentralAir',
    'Electrical',
    'Functional',
    'GarageType',
    'PavedDrive',
    'PoolQC',
    'MiscFeature',
    'SaleType',
    'SaleCondition',
    'MSSubClass']


QUAL_MAPPINGS = {'Po': 1, 'Fa': 2, 'TA': 3,
                 'Gd': 4, 'Ex': 5, 'Missing': 0, 'NA': 0}

EXPOSURE_MAPPINGS = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

FINISH_MAPPINGS = {'Missing': 0, 'NA': 0, 'Unf': 1,
                   'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

GARAGE_MAPPINGS = {'Missing': 0, 'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}

FENCE_MAPPINGS = {'Missing': 0, 'NA': 0,
                  'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}


# Variáveis ordinais (ordem conhecida)
ORDINAL_VARS = (
    QUAL_VARS +
    EXPOSURE_VARS +
    FINISH_VARS +
    GARAGE_VARS +
    FENCE_VARS
)

# Variáveis nominais (sem ordem)
NOMINAL_VARS = [v for v in CATEGORICAL_VARS
                if v not in ORDINAL_VARS and v != "MSSubClass"]



#############################################################
#### Pipeline
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
    
    ('yeojohnson', YeoJohnsonTransformer(variables=NUMERICALS_YEO_VARS)),
    
    ('binarizer', SklearnTransformerWrapper(
        transformer=Binarizer(threshold=0), variables=BINARIZE_VARS)),
    
    
    # Garante que MSSubClass continue sendo categórica
    ('cast_object',
     pp.CastVariablesAsObject(
         variables=['MSSubClass']
     )),


    # == CATEGORICAL ENCODING
    ('rare_label_encoder', RareLabelEncoder(
        tol=0.01, n_categories=1, variables=NOMINAL_VARS
    )),

    # === mappers ===
    ('mapper_qual', pp.Mapper(
        variables=QUAL_VARS, mappings=QUAL_MAPPINGS)),

    ('mapper_exposure', pp.Mapper(
        variables=EXPOSURE_VARS, mappings=EXPOSURE_MAPPINGS)),

    ('mapper_finish', pp.Mapper(
        variables=FINISH_VARS, mappings=FINISH_MAPPINGS)),

    ('mapper_garage', pp.Mapper(
        variables=GARAGE_VARS, mappings=GARAGE_MAPPINGS)),
    
    ('mapper_fence', pp.Mapper(
        variables=FENCE_VARS, mappings=FENCE_MAPPINGS)),



    # encode categorical and discrete variables using the target mean
    ('categorical_encoder', OrdinalEncoder(
        encoding_method='ordered', variables=NOMINAL_VARS)),
    
    ('mssubclass_encoder', OrdinalEncoder(
        encoding_method='ordered', variables=['MSSubClass'] )),
    
    
    
    
    
    
    ])


## entrenar o pipeline    
price_pipe.fit(X_train, y_train)

## aplicar transformacion de datos
X_train = price_pipe.transform(X_train)
X_test = price_pipe.transform(X_test)
