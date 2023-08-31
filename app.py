# %%
import streamlit as st
import pandas as pd

from pyspark.sql import functions as F

from lib import utils

print("Buscando modelo no MLFLOW...")
model = utils.get_model()
print("ok.")

print("Abrindo sessão spark...")
spark = utils.get_spark_session()
print("ok.")

print("Buscando times elegíveis...")
teams = utils.get_teams(spark)
print("ok.")

radiant_team_box = st.selectbox("Radiant", options=teams['descTeamName'])
dire_team_box = st.selectbox("Dire", options=teams['descTeamName'])

radiant_team_id = teams[teams['descTeamName']==radiant_team_box]['idTeam'].iloc[0]
dire_team_id = teams[teams['descTeamName']==dire_team_box]['idTeam'].iloc[0]

team_ids = [radiant_team_id, dire_team_id]

try:
    print("Buscando dados no database local...")
    fs_teams = utils.get_teams_database_fs(team_ids)
    print("ok.")

except:
   fs_teams = pd.DataFrame() 

if fs_teams.shape[0] < 2:
    print("Coletando dados do Databricks...")
    fs_teams = utils.get_teams_databricks_fs(spark, team_ids)
    print("ok")

    print("Inserindo dados no database local...")
    utils.insert_teams_database_fs(fs_teams)
    print("ok")

df_predict = utils.prepare_predict(fs_teams, radiant_team_id, dire_team_id)
dire_win, radiant_win = model.predict_proba(df_predict[model.feature_names_in_])[0]

df_winner = pd.DataFrame({
    "Radiant": [radiant_team_box],
    "Radiant Prob.": [radiant_win],
    "Dire": [dire_team_box],
    "Dire Prob.": [dire_win]
})

st.dataframe(df_winner)