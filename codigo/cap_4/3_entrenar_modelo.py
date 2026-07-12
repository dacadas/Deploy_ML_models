# para manipular os datasets
import pandas as pd
import numpy as np
import os
import random

# para plotagem de gráficos
import matplotlib.pyplot as plt

# para salvar o modelo
import joblib

# para construir o modelo
from sklearn.linear_model import Lasso

# para avaliar o modelo
from sklearn.metrics import mean_squared_error, r2_score

# para visualizar todas as colunas no dataframe
pd.pandas.set_option('display.max_columns', None)



def criar_pasta(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    
# Para fins de reprodutibilidade
semilla  = 42
np.random.seed( semilla )
random.seed(semilla) 


path_projecto = 'D:/Diego/Curso Udemy/Deploy_ML_models/dataset/'
path_data = path_projecto + '/dado_preprocesado/'
path_model = path_projecto + '/modelos/'

criar_pasta(path_model)




#### Carregar o dataset
X_train = pd.read_csv(path_data + 'xtrain.csv')
X_test  = pd.read_csv(path_data + 'xtest.csv')
y_train = pd.read_csv(path_data + 'ytrain.csv')
y_test  = pd.read_csv(path_data + 'ytest.csv')

## carregar as features selecionadas
features = pd.read_csv(path_data + 'selected_features.csv')
features = features['0'].to_list() 


## reduzir os conjuntos de treinamento pelas features selecionadas
X_train_sel = X_train[features]
X_test_sel  = X_test[features]


## Aplicar o modelo
lin_model     = Lasso(alpha=0.001, random_state=semilla)
lin_model_sel = Lasso(alpha=0.001, random_state=semilla)

# Treinar o modelo
lin_model.fit(X_train    , y_train)
lin_model_sel.fit(X_train_sel, y_train)

# Avaliar o modelo
# Para obter o verdadeiro desempenho do Lasso,
# temos que transformar tanto o alvo quanto as previsões,
# voltando aos valores originais dos preços das casas.

# Avaliaremos o desempenho utilizando o erro quadrático médio,
# a raiz do erro quadrático médio e o r2

pred_from_train = lin_model.predict(X_train)
pred_from_teste = lin_model.predict(X_test)

# determinar mse, rmse e r2
print('train mse: {}'.format(int( mean_squared_error(np.exp(y_train), np.exp(pred_from_train) ))))
print('train r2: {}'.format( r2_score(np.exp(y_train), np.exp(pred_from_train))))
print()
print('teste mse: {}'.format(int( mean_squared_error(np.exp(y_test), np.exp(pred_from_teste) ))))
print('teste r2: {}'.format( r2_score(np.exp(y_test), np.exp(pred_from_teste))))
print()


pred_from_train_sel = lin_model_sel.predict(X_train_sel)
pred_from_teste_sel = lin_model_sel.predict(X_test_sel)

# determinar mse, rmse e r2
print('train sel mse: {}'.format(int( mean_squared_error(np.exp(y_train), np.exp(pred_from_train_sel) ))))
print('train sel r2: {}'.format( r2_score(np.exp(y_train), np.exp(pred_from_train_sel))))
print()
print('teste sel mse: {}'.format(int( mean_squared_error(np.exp(y_test), np.exp(pred_from_teste_sel) ))))
print('teste sel r2: {}'.format( r2_score(np.exp(y_test), np.exp(pred_from_teste_sel))))
print()



# vamos avaliar nossas previsões em relação ao preço real de venda
plt.scatter(y_test, lin_model.predict(X_test))
plt.xlabel('True House Price')
plt.ylabel('Predicted House Price')
plt.title('Evaluation of Lasso Predictions')
plt.show()

# vamos avaliar nossas previsões em relação ao preço real de venda
plt.scatter(y_test, lin_model_sel.predict(X_test_sel))
plt.xlabel('True House Price')
plt.ylabel('Predicted House Price sel')
plt.title('Evaluation of Lasso Predictions sel')
plt.show()


# vamos avaliar a distribuição dos erros: 
# eles devem ter uma distribuição aproximadamente normal
y_test.reset_index(drop=True, inplace=True)
preds = pd.Series(lin_model.predict(X_test))

errors = y_test['SalePrice'] - preds
errors.hist(bins=30)
plt.show()


y_test.reset_index(drop=True, inplace=True)
preds_sel = pd.Series(lin_model_sel.predict(X_test_sel))

errors = y_test['SalePrice'] - preds_sel
errors.hist(bins=30)
plt.show()


# Finalmente, apenas por curiosidade, vamos olhar para a importância das features
importance = pd.Series(np.abs(lin_model.coef_.ravel()))
importance.index = X_train.columns
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(18,6))
plt.ylabel('Lasso Coefficients')
plt.title('Feature Importance')


importance = pd.Series(np.abs(lin_model_sel.coef_.ravel()))
importance.index = features
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(18,6))
plt.ylabel('Lasso Coefficients')
plt.title('Feature Importance sel')


### salvar
joblib.dump(lin_model,     path_model + 'linear_regression.joblib') 
joblib.dump(lin_model_sel, path_model + 'linear_regression_sel.joblib')