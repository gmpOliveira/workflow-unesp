# Databricks notebook source
# DBTITLE 1,Código para deixar em produção para estar pronto para novas partidas
import mlflow

model = mlflow.sklearn.load_model("models:/dota-unesp/production")

sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match_new")
df = sdf.toPandas()

# COMMAND ----------

features = list( set(df.columns.tolist()) - set(["match_id", "radiant_win"]))

x = df[features]

# COMMAND ----------

score = model.predict_proba(x)

df["proba_radiant_win"] = score[:,1]

df[["match_id", "radiant_win", "proba_radiant_win"]]

# COMMAND ----------


