# Databricks notebook source
# DBTITLE 1,Imports
# Import das libs
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier

import mlflow

# Import dos dados
sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match")

df = sdf.toPandas()

# COMMAND ----------

# DBTITLE 0,Definição das Variáveis
target_column = 'radiant_win'
id_column = 'match_id'

features_columns = list( set(df.columns.tolist()) - set([target_column,id_column]))

y = df[target_column]
x = df[features_columns]

x

# COMMAND ----------

# DBTITLE 1,Split Test e Train
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.2, random_state=42)

print("Número de linhas em x_train:", x_train.shape[0])
print("Número de linhas em x_test:", x_test.shape[0])
print("Número de linhas em y_train:", y_train.shape[0])
print("Número de linhas em y_test:", y_test.shape[0])

# COMMAND ----------

# DBTITLE 1,Setup do Experimento MLFlow
mlflow.set_experiment("/Users/gm.oliveira@unesp.br/dota-unesp-gm-oliveira")

# COMMAND ----------

# DBTITLE 1,Run do Experimento
with mlflow.start_run():
    
    mlflow.sklearn.autolog()

    # model = tree.DecisionTreeClassifier()

    #model = ensemble.AdaBoostClassifier(n_estimators=100, learning_rate=0.7)
    model = MLPClassifier()
    
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict(x_train)
    y_train_prob = model.predict_proba(x_train)

    acc_train = metrics.accuracy_score(y_train, y_train_pred)

    print("Acuracia em treino:", acc_train)
    
    y_test_pred = model.predict(x_test)
    y_test_prob = model.predict_proba(x_test)

    acc_test = metrics.accuracy_score(y_test, y_test_pred)
    print("Acuracia em teste:", acc_test)

# COMMAND ----------


