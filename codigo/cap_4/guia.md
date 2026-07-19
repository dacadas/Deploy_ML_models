# 📘 O MANUAL DO CIENTISTA DE DADOS: DA EXPLORAÇÃO AO PRÉ-PROCESSAMENTO

---

## PARTE 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)

> **Mentalidade desta fase:** O objetivo não é limpar ou alterar os dados estruturais ainda, mas entender o comportamento, a anatomia e as falhas do seu conjunto de dados brutos antes de tomar qualquer decisão regulatória.

---

### Passo 1: Investigar a Anatomia do Alvo (Target)

Antes de olhar para as causas, olhe para o efeito. A variável alvo dita o comportamento estatístico do modelo.

* Avalie as estatísticas descritivas e a distribuição visual do seu alvo
* Identifique se a distribuição é simétrica ou assimétrica (*skewed*)
* Se for muito assimétrica, uma transformação logarítmica pode ajudar

```python id="p1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(df['target'].describe())

df['target'].hist(bins=50)
plt.title('Distribuição Original do Alvo')
plt.show()

np.log(df['target']).hist(bins=50)
plt.title('Distribuição com Transformação Logarítmica')
plt.show()
```

💡 **Dica de Mestre:** Se o alvo tiver cauda longa à direita, o log estabiliza variância e melhora modelos lineares.

---

### Passo Extra: Analisar o Target em Relação às Variáveis

Entender o comportamento do alvo em relação às variáveis revela poder preditivo.

#### 🔸 Categóricas

```python id="p2"
print(df.groupby('coluna_categorica')['target'].mean().sort_values())

df.groupby('coluna_categorica')['target'].mean().plot(kind='bar')
```

#### 🔸 Numéricas

```python id="p3"
import seaborn as sns

sns.scatterplot(x=df['coluna_numerica'], y=df['target'])

df['faixa'] = pd.qcut(df['coluna_numerica'], q=10, duplicates='drop')
df.groupby('faixa')['target'].mean().plot(kind='bar')
```

💡 **Para classificação binária:**

```python id="p4"
print(df.groupby('coluna_categorica')['target_binario'].mean())
```

💡 **Insight:** Se o target muda entre grupos, a variável tem valor preditivo.

---

### Passo 2: A Divisão da Tribo (Tipagem de Variáveis)

* Separe categóricas e numéricas
* Inclua categorias “disfarçadas” (IDs, códigos)
* Converta para tipo `'O'`
* Separe discretas e contínuas
* Identifique temporais

```python id="p5"
cat_vars = [col for col in df.columns if df[col].dtype == 'O']

cat_vars = cat_vars + ['MSSubClass']
df[cat_vars] = df[cat_vars].astype('O')

num_vars = [col for col in df.columns if col not in cat_vars and col != 'target']

discrete_vars = [col for col in num_vars if len(df[col].unique()) < 20]
cont_vars = [col for col in num_vars if col not in discrete_vars]

year_vars = [col for col in num_vars if 'Year' in col or 'Ano' in col or 'Data' in col]
```

---

### Passo 3: Mapear os "Buracos" (Dados Faltantes)

* Identifique colunas com NA
* Calcule proporção
* Crie indicador
* Teste impacto no target

```python id="p6"
vars_with_na = [col for col in df.columns if df[col].isnull().sum() > 0]

print(df[vars_with_na].isnull().mean().sort_values(ascending=False))

tmp = df.copy()
tmp['na_indicador'] = np.where(tmp['coluna_teste'].isnull(), 1, 0)

print(tmp.groupby('na_indicador')['target'].mean())
```

💡 **Dica de Mestre:** Se o target muda com ausência, isso é informação.

💡 **Insight extra:** Missing pode representar comportamento, não erro.

---

### Passo 4: Detectar Raridades e Assimetrias Extremas

```python id="p7"
proporcoes = df['coluna_categorica'].value_counts() / len(df)
raras = proporcoes[proporcoes < 0.01].index

zeros_pct = (df['coluna_numerica'] == 0).mean()
```

---

### Passo Extra: Tratar Variáveis Muito Assimétricas

```python id="p8"
skewness = df[cont_vars].skew().sort_values(ascending=False)

skewed_vars = skewness[skewness > 1].index

for var in skewed_vars:
    df[var] = np.log1p(df[var])
```

💡 `log1p` evita erro com zero
💡 Yeo-Johnson funciona com negativos

---

### Passo Extra: Tratamento de Dados Temporais

⚠️ Nunca embaralhe dados temporais.

```python id="p9"
df = df.sort_values('data')

split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]
```
💡 Este tipo de divisão substitui o `train_test_split` para séries temporais.

#### Feature Engineering temporal

```python id="p10"
df['ano'] = df['data'].dt.year
df['mes'] = df['data'].dt.month
df['dia_semana'] = df['data'].dt.weekday

df['lag_1'] = df['target'].shift(1)
df['media_7'] = df['target'].rolling(window=7).mean()
```

💡 **Regra de ouro:** o futuro nunca pode influenciar o passado.

---

## PARTE 2: ENGENHARIA DE DADOS (FEATURE ENGINEERING)

> **Mentalidade desta fase:** Transformar dados sem vazamento de informação.

---

### Passo 1: A Blindagem Sagrada (Split Treino e Teste)

⚠️ **Erro comum:** usar estatísticas antes do split invalida o modelo.

```python id="p11"
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

X = df.drop(['target'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=SEED
)

y_train = np.log(y_train)
y_test = np.log(y_test)
```

---

### Passo 2: Imputação Inteligente de Dados Faltantes

```python id="p12"
# CATEGÓRICAS
X_train['coluna_cat'] = X_train['coluna_cat'].fillna('Missing')
X_test['coluna_cat'] = X_test['coluna_cat'].fillna('Missing')

moda = X_train['coluna_cat_pouco_na'].mode()[0]

X_train['coluna_cat_pouco_na'] = X_train['coluna_cat_pouco_na'].fillna(moda)
X_test['coluna_cat_pouco_na'] = X_test['coluna_cat_pouco_na'].fillna(moda)

# NUMÉRICAS
mediana = X_train['coluna_num'].median()

X_train['coluna_num_na'] = np.where(X_train['coluna_num'].isnull(), 1, 0)
X_test['coluna_num_na'] = np.where(X_test['coluna_num'].isnull(), 1, 0)

X_train['coluna_num'] = X_train['coluna_num'].fillna(mediana)
X_test['coluna_num'] = X_test['coluna_num'].fillna(mediana)
```

---

### Passo 3: Lapidação de Variáveis Numéricas e Temporais

```python id="p13"
import scipy.stats as stats

X_train['coluna_continua'] = np.log(X_train['coluna_continua'])
X_test['coluna_continua'] = np.log(X_test['coluna_continua'])

X_train['coluna_area'], lmbda = stats.yeojohnson(X_train['coluna_area'])
X_test['coluna_area'] = stats.yeojohnson(X_test['coluna_area'], lmbda=lmbda)

X_train['coluna_assimetrica'] = np.where(X_train['coluna_assimetrica'] == 0, 0, 1)
X_test['coluna_assimetrica'] = np.where(X_test['coluna_assimetrica'] == 0, 0, 1)

X_train['Tempo_Decorrido'] = X_train['Ano_Fim'] - X_train['Ano_Inicio']
X_test['Tempo_Decorrido'] = X_test['Ano_Fim'] - X_test['Ano_Inicio']
```

---

### Passo 4: Codificação de Categorias (Encoding Monotônico)

💡 Ordenação baseada no target melhora modelos lineares.

```python id="p14"
proporcao = X_train['coluna_texto'].value_counts() / len(X_train)
freq = proporcao[proporcao >= 0.01].index

X_train['coluna_texto'] = np.where(
    X_train['coluna_texto'].isin(freq),
    X_train['coluna_texto'],
    'Rare'
)

X_test['coluna_texto'] = np.where(
    X_test['coluna_texto'].isin(freq),
    X_test['coluna_texto'],
    'Rare'
)

tmp = pd.concat([X_train, y_train], axis=1)

ordenadas = tmp.groupby('coluna_texto')['target'].mean().sort_values().index

mapa = {cat: i for i, cat in enumerate(ordenadas)}

X_train['coluna_texto'] = X_train['coluna_texto'].map(mapa)
X_test['coluna_texto'] = X_test['coluna_texto'].map(mapa)
```

---

### Passo 5: Escalonamento Final (Scaling) e Exportação

⚠️ Nunca ajustar scaler no dataset completo.

```python id="p15"
from sklearn.preprocessing import MinMaxScaler
import joblib

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

X_train.to_csv('X_train_final.csv', index=False)
X_test.to_csv('X_test_final.csv', index=False)

joblib.dump(scaler, 'minmax_scaler.joblib')
```

---

## ⚠️ Erros Comuns

* Fazer feature engineering antes do split
* Usar estatísticas globais (*data leakage*)
* Ignorar categorias raras
* Não tratar assimetria
* Aplicar scaling antes da divisão
* Em dados temporais: misturar passado e futuro

---

## ✅ Conclusão

Seguindo esse fluxo você garante:

* Dados bem compreendidos
* Features robustas
* Pipeline reproduzível
* Modelos mais estáveis



---

# PARTE 3: PIPELINES E PRODUÇÃO

> **Mentalidade desta fase:** Tudo que você fez manualmente agora precisa ser reproduzível, escalável e à prova de erro. Se não está em um pipeline, não está pronto para produção.

---

💡 Tudo que foi feito manualmente na Parte 2 agora será automatizado dentro do pipeline.

💡 No pipeline, técnicas podem mudar (ex: OneHot vs encoding ordinal), mas o princípio permanece.

### Passo 1: Por que usar Pipeline?

Até agora você executou várias etapas manualmente:

* Tratamento de missing
* Transformações
* Encoding
* Scaling

⚠️ Problema:

* Alto risco de erro humano
* Difícil reprodução
* Não escala
* Propenso a *data leakage*

💡 **Objetivo do pipeline:** encapsular todo o fluxo em um único objeto treinável.

---

### Passo 2: Separando Tipos de Variáveis
💡 A partir daqui, assumimos que os dados já estão separados em `X` e `y`.

💡 assumir que X já não contém o target

```python id="a1"
cat_vars = [col for col in X.columns if X[col].dtype == 'O']
num_vars = [col for col in X.columns if col not in cat_vars]
```

---


### Passo 3: Pipeline Numérico

```python id="a2"
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])
```

---

### Passo 4: Pipeline Categórico

```python id="a3"
from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
```

---

### Passo 5: ColumnTransformer

```python id="a4"
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_vars),
    ('cat', cat_pipeline, cat_vars)
])
```

---

### Passo 6: Pipeline Completo

```python id="a5"
from sklearn.linear_model import LinearRegression

model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', LinearRegression())
])
```

---

### Passo 7: Treinamento e Previsão

```python id="a6"
model.fit(X_train, y_train)

preds = model.predict(X_test)
```

---

## 🔥 Passo 8: Criando Transformações Customizadas

Até aqui você usou apenas transformações prontas.

👉 Mas no mundo real, isso **não é suficiente**.

Você precisa encapsular lógica como:

* log transform seletivo
* criação de variáveis
* tratamento de skew
* regras de negócio

---

### Quando usar um Transformer customizado?

Use quando você tem lógica que:

* Não existe no sklearn
* Depende do seu dataset
* Precisa ser reproduzida em produção
* Precisa evitar *data leakage*

👉 Exemplos clássicos:

* aplicar `log` só em colunas específicas
* criar variável de tempo
* agrupar categorias raras
* criar flags (ex: missing, zero, etc.)

---

### Estrutura básica de um Transformer

```python id="a7"
from sklearn.base import BaseEstimator, TransformerMixin

class MeuTransformer(BaseEstimator, TransformerMixin):
	#`log1p` evita erro com zero
    def __init__(self, variaveis):
        self.variaveis = variaveis

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        for var in self.variaveis:
            X[var] = np.log1p(X[var])
            
        return X
```

---

### 💡 Entendendo o padrão

* `BaseEstimator` → permite integração com sklearn
* `TransformerMixin` → adiciona `.fit_transform()` automaticamente

---

### 💡 Regra de ouro

* `fit()` → aprende parâmetros (ex: média, categorias)
* `transform()` → aplica transformação

👉 Nunca misture os dois.

---

### Exemplo 1: Log Transformer

```python id="a8"
class LogTransformer(BaseEstimator, TransformerMixin):
    #`log1p` evita erro com zero
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for var in self.variables:
            X[var] = np.log1p(X[var])

        return X
```

---

### Exemplo 2: Rare Label Encoder

```python id="a9"
class RareLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.01, variables=None):
        self.tol = tol
        self.variables = variables
        self.frequent_labels = {}

    def fit(self, X, y=None):
        X = X.copy()

        for var in self.variables:
            freq = X[var].value_counts(normalize=True)
            self.frequent_labels[var] = freq[freq >= self.tol].index

        return self

    def transform(self, X):
        X = X.copy()

        for var in self.variables:
            X[var] = np.where(
                X[var].isin(self.frequent_labels[var]),
                X[var],
                'Rare'
            )

        return X
```

---

### Exemplo 3: Time Features

```python id="a10"
class TimeFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variable):
        self.variable = variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['ano'] = X[self.variable].dt.year
        X['mes'] = X[self.variable].dt.month
        X['dia_semana'] = X[self.variable].dt.weekday

        return X
```

---

## 🔗 Passo 9: Usando Transformers no Pipeline

```python id="a11"
model = Pipeline(steps=[
    ('log', LogTransformer(variables=['coluna1', 'coluna2'])),
    ('rare', RareLabelEncoder(variables=cat_vars)),
    ('preprocessing', preprocessor),
    ('model', LinearRegression())
])
```

---

## ⚠️ Pontos Críticos

* Nunca transformar dados fora do pipeline
* Nunca usar `.fit()` no teste
* Sempre separar lógica de `fit` e `transform`
* Manter consistência no tipo de saída (DataFrame ou array)

---

## 💡 Dica de Mestre

Se sua feature engineering está fora do pipeline...

👉 você ainda está em modo “notebook”, não em produção.

---

## 🧠 Conexão Final

* Parte 1 → entender
* Parte 2 → transformar
* Parte 3 → automatizar

👉 Isso fecha o ciclo completo de um projeto profissional de Data Science.

---

# PARTE 4: ENGENHARIA AVANÇADA E PIPELINES INDUSTRIAIS

> **Mentalidade desta fase:** Sair de pipelines genéricos e construir esteiras de dados robustas, reproduzíveis e alinhadas às regras de negócio. Aqui você deixa de apenas usar ferramentas e passa a projetar sistemas de ML.

---

## 💡 Conexão com as Partes Anteriores

* Parte 1 → entender os dados
* Parte 2 → transformar manualmente
* Parte 3 → automatizar com pipeline
* **Parte 4 → industrializar e customizar profundamente**

---

### Passo 1: Imputação Inteligente e Estado Estatístico

No mundo real, imputação simples não é suficiente.

👉 Você precisa:

* Preservar o comportamento de missing
* Separar estratégias por tipo de variável
* Garantir que o estado aprendido venha **apenas do treino**

---

#### Estratégia robusta

```python id="b1"
from feature_engine.imputation import (
    CategoricalImputer,
    MeanMedianImputer,
    AddMissingIndicator
)

# categóricas com muitos NA
('missing_imputation', CategoricalImputer(
    imputation_method='missing',
    variables=CATEGORICAL_VARS_WITH_NA_MISSING
)),

# categóricas com poucos NA
('frequent_imputation', CategoricalImputer(
    imputation_method='frequent',
    variables=CATEGORICAL_VARS_WITH_NA_FREQUENT
)),

# indicador de ausência
('missing_indicator', AddMissingIndicator(
    variables=NUMERICAL_VARS_WITH_NA
)),

# numéricas
('mean_imputation', MeanMedianImputer(
    imputation_method='mean',
    variables=NUMERICAL_VARS_WITH_NA
))
```

💡 **Dica de Mestre:** Missing raramente é aleatório. Preserve essa informação.

---

#### Alternativa: Transformer customizado

```python id="b2"
class MeanImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        self.imputer_dict_ = X[self.variables].mean().to_dict()
        return self

    def transform(self, X):
        X = X.copy()

        for var in self.variables:
            X[var] = X[var].fillna(self.imputer_dict_[var])

        return X
```

👉 O segredo: **estado aprendido no `fit()` e congelado**

---

### Passo 2: Regras de Negócio com Mapeamento Explícito

Nem tudo deve ser aprendido pelo modelo.

👉 Muitas variáveis já têm ordem definida pelo negócio.

```python id="b3"
class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables, mappings):
        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for var in self.variables:
            X[var] = X[var].map(self.mappings)

        return X
```

---

#### Exemplo

```python id="b4"
QUAL_MAPPINGS = {
    'Po': 1, 'Fa': 2, 'TA': 3,
    'Gd': 4, 'Ex': 5,
    'Missing': 0, 'NA': 0
}
```

💡 **Insight:** Isso evita que o modelo “reinvente” regras já conhecidas.

---

### Passo 3: Engenharia Temporal Correta

Datas absolutas são fracas para modelos.

👉 O correto é usar **tempo relativo**

```python id="b5"
class TemporalVariableTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables, reference_variable):
        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for var in self.variables:
            X[var] = X[self.reference_variable] - X[var]

        return X
```

---

#### Uso no pipeline

```python id="b6"
('elapsed_time', TemporalVariableTransformer(
    variables=TEMPORAL_VARS,
    reference_variable=REF_VAR
))
```

⚠️ **Regra de ouro:** remova a variável de referência depois

---

### Passo 4: Encoding Monotônico Guiado pelo Target

Para categorias complexas:

👉 ordenar categorias pelo comportamento do target

```python id="b7"
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        self.encoder_dict_ = {}

        for var in self.variables:
            ordenacao = (
                temp.groupby(var)["target"]
                .mean()
                .sort_values()
                .index
            )

            self.encoder_dict_[var] = {
                k: i for i, k in enumerate(ordenacao)
            }

        return self

    def transform(self, X):
        X = X.copy()

        for var in self.variables:
            X[var] = X[var].map(self.encoder_dict_[var])

        return X
```

💡 **Quando usar:**

* modelos lineares
* evitar explosão de dimensões (OneHot)

---

### Passo 5: Blindagem de Tipagem

Algumas variáveis são numéricas mas **não são quantitativas**.

```python id="b8"
class CastVariablesAsObject(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for var in self.variables:
            X[var] = X[var].astype('O')

        return X
```

💡 Exemplo: `MSSubClass`

---

### Passo 6: Pipeline Industrial Consolidado

Agora unificamos tudo.

```python id="b9"
from sklearn.pipeline import Pipeline
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
from feature_engine.transformation import LogTransformer, YeoJohnsonTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import Binarizer

pipeline = Pipeline([

    # 1. IMPUTAÇÃO
    ('missing', CategoricalImputer(imputation_method='missing',
                                  variables=CATEGORICAL_VARS_WITH_NA_MISSING)),

    ('mean', MeanMedianImputer(imputation_method='mean',
                              variables=NUMERICAL_VARS_WITH_NA)),

    # 2. TEMPORAL
    ('time', TemporalVariableTransformer(
        variables=TEMPORAL_VARS,
        reference_variable=REF_VAR)),

    # 3. TRANSFORMAÇÕES
    ('log', LogTransformer(variables=NUMERICALS_LOG_VARS)),
    ('yeo', YeoJohnsonTransformer(variables=NUMERICALS_YEO_VARS)),

    # 4. REGRAS DE NEGÓCIO
    ('mapper', Mapper(variables=QUAL_VARS, mappings=QUAL_MAPPINGS)),

    # 5. CATEGÓRICAS
    ('rare', RareLabelEncoder(tol=0.01, variables=NOMINAL_VARS)),
    ('encoder', OrdinalEncoder(encoding_method='ordered',
                               variables=NOMINAL_VARS)),
])
```

---

### Treinamento seguro

```python id="b10"
pipeline.fit(X_train, y_train)

X_train = pipeline.transform(X_train)
X_test = pipeline.transform(X_test)
```

---

## ⚠️ Erros Críticos Desta Fase

* Calcular parâmetros no `transform()`
* Não usar `X.copy()`
* Misturar regras de negócio com estatística sem controle
* Não remover variáveis redundantes
* Quebrar tipagem categórica

---

## 💡 Dica de Mestre

Se seu pipeline não consegue:

```python id="b11"
pipeline.predict(dado_novo)
```

sem nenhuma intervenção manual…

👉 ele ainda não está pronto para produção.

---

## 🧠 Conclusão Final do Guia

Você evoluiu de:

1. Explorar dados
2. Criar features
3. Automatizar
4. **Construir sistemas de dados robustos**

👉 Isso é o ciclo completo de um cientista de dados profissional.




Seleção de Variáveis dentro do Pipeline

A decisão de incluir a seleção de variáveis (Feature Selection) dentro do pipeline depende do contexto do projeto.

✅ Vantagens
Não é necessário codificar manualmente quais variáveis serão usadas pelo modelo.
Facilita o retreinamento do modelo, pois toda a engenharia de atributos e seleção já fazem parte do pipeline.
Ideal quando o modelo é atualizado frequentemente utilizando o mesmo conjunto de dados.
❌ Desvantagens
É preciso implementar e manter o código de engenharia de todas as variáveis, inclusive das que serão descartadas pela seleção.
Aumenta a quantidade de código, testes, tratamento de erros e manutenção.
Torna o pipeline menos flexível para receber novas variáveis ou novas fontes de dados, pois será necessário reconstruir a pipeline.
Quando vale a pena incluir a seleção de variáveis no pipeline?

✔ Quando:

O modelo será retreinado frequentemente.
O conjunto de dados permanece praticamente o mesmo.
O número de variáveis é relativamente pequeno ou moderado.

❌ Evite quando:

O conjunto possui muitas variáveis (grande espaço de atributos).
Novas fontes de dados são adicionadas com frequência.
As variáveis utilizadas pelo modelo mudam constantemente.
Dicas práticas
Projetos pequenos ou estáveis: inclua a seleção de variáveis no pipeline. Simplifica o treinamento e o deploy.
Projetos grandes ou em evolução: faça a seleção de variáveis durante o desenvolvimento e implante apenas as variáveis realmente utilizadas.
Sempre considere o custo de manutenção da pipeline, não apenas o desempenho do modelo.
Quanto mais etapas de engenharia de atributos existirem, maior será o esforço para testar, documentar e manter o código.

Regra prática: se o modelo será atualizado várias vezes com os mesmos dados, faz sentido incluir a seleção de variáveis no pipeline. Se os dados e as variáveis mudam com frequência, é mais simples realizar a seleção antes da implantação e manter a pipeline apenas com as transformações necessárias.